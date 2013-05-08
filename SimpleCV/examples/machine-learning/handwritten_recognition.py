import numpy
import os
from pybrain import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from SimpleCV import download_and_extract, Image, LAUNCH_PATH

handwritten_digits = 'https://github.com/rsiera/datasets/raw/master/handwritten_dataset.tar.gz'
data_path = download_and_extract(handwritten_digits)
print 'Test Images Downloaded at:', data_path


def create_dataset_handwritten(test=None):
    dataset = ClassificationDataSet(784, 1, nb_classes=10)
    if test:
        samples_path = os.path.join(LAUNCH_PATH, 'sampleimages',
                                    'handwritten_samples')
        for fid in os.listdir(samples_path):
            out = fid.split('_')[0]
            img = Image(os.path.join(samples_path, fid))
            contour_array = img.getGrayNumpy().T.flatten()
            dataset.addSample(contour_array, [out])
            dataset.assignClasses()

    else:
        dataset_path = os.path.join(data_path, 'handwritten_dataset')
        for obj in os.listdir(dataset_path):
            example_digit = os.path.join(dataset_path, obj)
            digit = int(example_digit[-1].strip())
            with open(example_digit, 'rb') as fid:
                for x in xrange(500):
                    data_array = numpy.fromfile(fid, dtype=numpy.uint8,
                                                count=28 * 28)
                    dataset.addSample(data_array.tolist(), [digit])
            dataset.assignClasses()

    return dataset


training_dataset = create_dataset_handwritten()
training_dataset._convertToOneOfMany()

testing_dataset = create_dataset_handwritten(test=True)
testing_dataset._convertToOneOfMany()

hidden_neuron = 250
fnn = buildNetwork(training_dataset.indim, hidden_neuron,
                   training_dataset.outdim,
                   outclass=SoftmaxLayer)

# fnn = NetworkReader.readFrom(network_path)
trainer = BackpropTrainer(fnn, dataset=training_dataset, learningrate=0.05,
                          momentum=0.5, verbose=True)

for i in range(100):
    trainer.trainEpochs(15)

    # evaluate the result on the training and test data
    print 'train ans: ', trainer.testOnClassData(dataset=training_dataset,
                                                 verbose=True)
    print 'train corr: ', training_dataset['class'].transpose()[0]

    print 'test ans: ', trainer.testOnClassData(dataset=testing_dataset)
    print 'test corr: ', testing_dataset['class'].transpose()[0]

    trnresult = percentError(
        trainer.testOnClassData(dataset=training_dataset, verbose=True),
        training_dataset['class'])
    tstresult = percentError(trainer.testOnClassData(dataset=testing_dataset),
                             testing_dataset['class'])

    # print the result
    result = "epoch: %4d  train error: %5.2f%%  test error: %5.2f%%\n" % (
        trainer.totalepochs, trnresult, tstresult)
    print result

print 'train ans: ', trainer.testOnClassData(dataset=training_dataset,
                                             verbose=True)
trnresult = percentError(
    trainer.testOnClassData(dataset=training_dataset, verbose=True),
    training_dataset['class'])
