#Nur Für Visual der Progress bar und MAtplotlib
import datasets
datasets.logging.set_verbosity_error()

# disables the progress bar for notebooks: https://github.com/huggingface/datasets/issues/2651
datasets.logging.get_verbosity = lambda: logging.NOTSET

from matplotlib import rcParams
rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.labelsize': 16})


import torch
import numpy as np
import random
n = random.randint(0,5000)
seed = n #2022 default
torch.manual_seed(seed)
np.random.seed(seed)


#Dataset import
import logging
#raw_dataset = datasets.load_dataset('ag_news') #amazon-review-polarity #original 'rotten_tomatoes' #trec für trec-6 auf paper??gehtnicht
raw_dataset = datasets.load_dataset('ag_news') #falls trec zeilen drunter entkommentieren
#raw_dataset = raw_dataset.rename_column("label-coarse", "label")
#raw_dataset = raw_dataset.rename_column("content", "text") #für amazon_polarity





#Trec hat 2 art von labels, wollen den einfaches trec-6 label satz


num_classes = np.unique(raw_dataset['train']['label']).shape[0]

print('First 3 training samples:\n')
for i in range(3):
    print(raw_dataset['train']['label'][i], ' ', raw_dataset['train']['text'][i])


#small text brauch TransformersDataset format
#bert-base-uncased ist dabei das transformer model

from transformers import AutoTokenizer

transformer_model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(
    transformer_model_name
)


#get_transformers_dataset ist die helper function, welche tokenizer.encode_plus() benutzt um dann 
#TransformersDataset instanz zu erstellen

from small_text.integrations.transformers.datasets import TransformersDataset

def get_transformers_dataset(tokenizer, data, labels, max_length=60):

    data_out = []

    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )

        data_out.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i]))

    #print("-sd",data_out[1])
    return TransformersDataset(data_out)

#print(raw_dataset['train']['text'][:10])
#print(raw_dataset['train']['label'][:10])

train = get_transformers_dataset(tokenizer, raw_dataset['train']['text'], raw_dataset['train']['label'])
test = get_transformers_dataset(tokenizer, raw_dataset['test']['text'], raw_dataset['test']['label'])

#Bauen eine PoolBasedActiveLearner instanz folgendes braucht:
#classifier factory
#query strategy und train dataset


from small_text.active_learner import PoolBasedActiveLearner

from small_text.initialization import random_initialization_balanced
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.query_strategies import DeepEnsamble,TrustScore2,TrustScore,EvidentialConfidence2,BT_Temp,TemperatureScaling,BreakingTies,RandomSampling,PredictionEntropy,FalkenbergConfidence2,FalkenbergConfidence,LeastConfidence
#nicht vergessen bei neuer strategie im query strategie ordner beim init das hinnzuzufügen
from small_text.integrations.transformers import TransformerModelArguments


# simulates an initial labeling to warm-start the active learning process
def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_balanced(y_train, n_samples=25)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial



transformer_model = TransformerModelArguments(transformer_model_name)
clf_factory = TransformerBasedClassificationFactory(transformer_model, 
                                                    num_classes, 
                                                    kwargs=dict({'device': 'cuda',#cpu/cuda geht auch 
                                                                 'mini_batch_size': 64,
                                                                 'class_weight': 'balanced'
                                                                }))

#Die Zeile ist einmal CLS Vektor umwandeln, damit man es nicht jedes mal machen muss für FalkenbergConfidence

query_strategy = PredictionEntropy()#BT_Temp() #PredictionEntropy() #RandomSampling() #BreakingTies() #RandomSampling() FalkenbergConfidence() LeastConfidence#
#query_strategy = RandomSampling() #zeile wo falkenbergCLS übergeben wird löschen

active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train) #train ist dataset 

indices_labeled = initialize_active_learner(active_learner, train.y)

#Wird gebraucht für FalkenbergConfidence und EvidentialConfidence, nicht rechenintensiv also wird einfach gemacht
active_learner._query_strategy.num_classes = active_learner._clf_factory.num_classes
active_learner._query_strategy.factory = clf_factory # nur für DeepEnsamble nötig

"""
#-------------------------------------------------------------NUR NÖTIG FÜR FALKENBERGCONFIDENCE UND,TrustScore
print("Einmalige CLS Vektor Erstellung der Trainingsdaten, NUR BEI EVIDENT UND FALKENBERG NÖTIG")
#DIESEN PART AUSKOMMENTIEREN WENN NICHT FALKENBERGCONFIDENCE GEMACHT WIRD
unlabeledList = []
#for abc in _indices_unlabeled:
for abc in range(len(train.x)):
    traindata =train.data[abc]
    unlabeledList.append((traindata[0],traindata[1],traindata[2]))
transformerDataUNLABELED = TransformersDataset(unlabeledList)
_clsUNLABELED = active_learner._clf.embed(transformerDataUNLABELED,embedding_method='cls')

active_learner._query_strategy._clsUNLABELED = _clsUNLABELED
print("Einmalige CLS Vektor Erstellung Efolgreich:",float(_clsUNLABELED[0][0]) is not float(0),_clsUNLABELED[0][0])
"""


#Hier wird für Evidential noch der Eval loader gemacht, nicht zum training benutzt nur zur Reference 
#Einfachhaltshalber sind es alle trainingsdaten mit label
"""
from torch.utils.data import TensorDataset,DataLoader

my_dataset = TensorDataset(torch.from_numpy(_clsUNLABELED),torch.tensor(raw_dataset['train']['label'])) # create your datset
evalLoader = DataLoader(my_dataset, batch_size=1000, shuffle=True, num_workers=1) # create your dataloader
active_learner._query_strategy.evalLoader = evalLoader

#------------------------------------------------------------------------------------------------------
"""

#print("CHECK")
#print(train.data[-4:])
#print()
#print(_clsUNLABELED[-4:])




#main active learning loop, declares welche docs als nächstes gelabeled werden
#weil wir die labels schon haben geben wir sie direkt weiter 

#in echt müssten die labels von menschen gegeben werden und test accuracy könnte nicht abgeschätz werden 

from sklearn.metrics import accuracy_score


num_queries = 20 #original 10, für cpu ~5mal langsamer hier, 20 für testwerte

def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)
    
    test_acc = accuracy_score(y_pred_test, test.y)

    print('Train accuracy: {:.4f}'.format(accuracy_score(y_pred, train.y)))
    print('Test accuracy: {:.4f}'.format(test_acc))
    
    return test_acc


results = []
results.append(evaluate(active_learner, train[indices_labeled], test))



"""
###Custom TransformerDataSet from Labeled Date -> Embedded CLS VEKTOR, hier ohne embedding

finaloutput = []
for a in active_learner.indices_labeled:

    finaloutput.append((train.data[a][0],train.data[a][1],train.data[a][2]))
customTransformer = TransformersDataset(finaloutput)

print(customTransformer.data[0])
"""

#tester = active_learner._clf.embed(active_learner.dataset,embedding_method='cls') #transformer classification 120+
#print("143",tester)
#print("143",tester.shape)
#print("----")


for i in range(num_queries):
    #bevor es nächsten queried oder im query befehl soll es dann CLS Vektor mit CustomTransset machen



    # ...where each iteration consists of labelling 20 samples
    indices_queried = active_learner.query(num_samples=25)

    # Simulate user interaction here. Replace this for real-world usage.
    y = train.y[indices_queried]

    # Return the labels for the current query to the active learner.
    active_learner.update(y)

    indices_labeled = np.concatenate([indices_queried, indices_labeled])
    
    #print('---------------')
    print(f'Iteration #{i} ({len(indices_labeled)} samples)')
    results.append(evaluate(active_learner, train[indices_labeled], test))



f = open("logs.txt", "a")
print("Results hier---")
f.write("\n Neuer Run ------------\n")
for x2 in results:
    f.write(str(x2)+ "\n")
    print(x2)

f.close()

"""
#viz

import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
ax = plt.axes()

data = np.vstack((np.arange(num_queries+1), np.array(results)))
sns.lineplot(x=0, y=1, data=data)

plt.xlabel('number of queries', labelpad=15)
plt.ylabel('test accuracy', labelpad=25)

sns.despine()

plt.savefig('smallText.png')
"""
