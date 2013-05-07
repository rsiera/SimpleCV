#!/usr/bin/env python
#-*- encoding: utf8 -*-
# vim: set ts=4 sw=4 fdm=indent : */
from itertools import product

import cv
import os
from pybrain import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml import NetworkReader, NetworkWriter
from pybrain.utilities import percentError
from SimpleCV import Image

ROOT_PATH = os.getcwd()
NN_PATH = os.path.join(ROOT_PATH, 'dataset')

PATHS_TYPE = {
    'TRAINING': os.path.join(NN_PATH, 'training'),
    'VALIDATE': os.path.join(NN_PATH, 'valid'),
    'TESTING': os.path.join(NN_PATH, 'test')
}


class NN(object):
    def __init__(self, learning_rate, momentum, epochs, fnn_path=None,
                 cgm_path=None, all_epochs=100,
                 hidden_neuron=128):
        self.hidden_neuron = hidden_neuron
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.all_epochs = all_epochs
        self.fnn_path = fnn_path
        self.cgm_path = cgm_path
        self.tmp_img = None
        self.data = []
        self.fnn = self.build_fnn()
        self.cgm = self.build_cgm()

    def __repr__(self):
        for mod in self.fnn.modules:
            print "Module:", mod.name
            if mod.paramdim > 0:
                print "--parameters:", mod.params
            for conn in self.fnn.connections[mod]:
                print "-connection to", conn.outmod.name
                if conn.paramdim > 0:
                    print "- parameters", conn.params
            if hasattr(self.fnn, "recurrentConns"):
                print "Recurrent connections"
                for conn in self.fnn.recurrentConns:
                    print "-", conn.inmod.name, " to", conn.outmod.name
                    if conn.paramdim > 0:
                        print "- parameters", conn.params

    def read_structures(self):
        self.fnn = NetworkReader.readFrom(self.fnn_path)
        self.cgm = NetworkReader.readFrom(self.cgm_path)

    def run_sliding_window(self, img_data, res=10, numpc=14):
        """
        with sliding window method
        """
        self.read_structures()
        self.tmp_img = Image(img_data)
        list_window = self.tmp_img.inc_recur_window(90, 66)

        for window_size in list_window:
            for y, x in product(
                xrange(self.tmp_img.size_img[1] - window_size[1], res),
                xrange(self.tmp_img.size_img[0] - window_size[0]), res):
                box = (x, y, window_size[0], window_size[1])
                cv.SetImageROI(self.tmp_img, box)

                res_image = self.tmp_img.resize(40, 30)
                contour_array, edge_area = Image.canny(res_image)

                if edge_area >= 0.11:
                    result_fnn = self.fnn.activate(contour_array)

                    if result_fnn.argmax() == 1.0:
                        result_cgm = self.cgm.activate(contour_array)

                        if result_cgm.argmax() <= 10e-06:
                            cv.Rectangle(self.tmp_img, (x, y),
                                         (x + window_size[0],
                                          y + window_size[1]),
                                         cv.RGB(0, 255, 0), 1, 8, 0)

        cv.SaveImage(os.path.join(ROOT_PATH, '%s.png' % self.tmp_img.filename),
                     self.tmp_img)

    def create_dataset(self, **kwargs):
        dataset = ClassificationDataSet(1200, 1, nb_classes=2,
                                        class_labels=['False', 'True'])
        files_path = PATHS_TYPE[kwargs['type_data']]

        for img_path in os.listdir(files_path):
            res = img_path.split('.')[0][-1]

            img_full_path = os.path.join(files_path, img_path)
            img = Image(img_full_path)
            self.data.append(Image(img_full_path))

            contour_array = []
            for y, x in product(xrange(img.size_img[1]),
                                xrange(img.size_img[0])):
                c = cv.Get2D(img.cv_img, y, x)
                coordinate = 1 if c[0] == 255.0 else -1
                contour_array.append(coordinate)
            dataset.addSample(contour_array, [res])

        dataset.assignClasses()
        return dataset

    def build_fnn(self):
        training_dataset = self.create_dataset(type_data='TRAINING')
        training_dataset._convertToOneOfMany(bounds=[0, 1])

        print 'statistics', training_dataset.calculateStatistics()
        print training_dataset.getClass(1)

        testing_dataset = self.create_dataset(type_data='VALIDATE')
        testing_dataset._convertToOneOfMany(bounds=[0, 1])

        self.fnn = buildNetwork(training_dataset.indim, self.hidden_neuron,
                                training_dataset.outdim, outclass=SoftmaxLayer)

        trainer = BackpropTrainer(self.fnn, dataset=training_dataset,
                                  learningrate=self.learning_rate,
                                  momentum=self.momentum, verbose=True)

        last_overloaded = 100

        for i in xrange(self.all_epochs):
            trainer.trainEpochs(self.epochs)

            print 'TRAIN_ANS: ', trainer.testOnClassData(
                dataset=training_dataset, verbose=True)
            print 'TRAINING_TP: ', training_dataset['class'].transpose()[0]

            print 'TESTING_ANS: ', trainer.testOnClassData(
                dataset=testing_dataset)
            print 'TESTING_TP: ', testing_dataset['class'].transpose()[0]

            trnresult = percentError(
                trainer.testOnClassData(dataset=training_dataset, verbose=True),
                training_dataset['class'])
            tstresult = percentError(
                trainer.testOnClassData(dataset=testing_dataset),
                testing_dataset['class'])

            if tstresult > last_overloaded:
                break

            last_overloaded = tstresult

            if tstresult <= 0.001:
                if not os.path.exists(os.path.join(NN_PATH, 'fnn')):
                    os.makedirs(os.path.join(NN_PATH, 'fnn'))

                result = "epoch: %4d  train err: %5.2f%% test err: %5.2f%%\n" % (
                    trainer.totalepochs, trnresult, tstresult)

                result_file = open(os.path.join(NN_PATH, 'fnn',
                    '%s_%s' % (self.hidden_neuron, trainer.totalepochs)), 'w')
                result_file.write(result)
                result_file.close()

                network_path = os.path.join(NN_PATH, 'fnn',
                    '%s_%s.xml' % (self.hidden_neuron, trainer.totalepochs))
                NetworkWriter.writeToFile(self.fnn, network_path)

    def build_cgm(self):
        training_dataset = self.create_dataset(type_data='TRAINING')
        training_dataset._convertToOneOfMany(bounds=[0, 1])

        testing_dataset = self.create_dataset(type_data='VALIDATE')
        testing_dataset._convertToOneOfMany(bounds=[0, 1])

        self.cgm = buildNetwork(training_dataset.indim, [100, 75],
                                training_dataset.indim, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(self.cgm, dataset=training_dataset,
            learningrate=self.learning_rate, momentum=self.momentum, verbose=True)

        last_overloaded = 100

        for i in xrange(self.all_epochs):
            trainer.trainEpochs(self.epochs)

            trnresult = percentError(
                trainer.testOnClassData(dataset=training_dataset, verbose=True),
                training_dataset['class'])
            tstresult = percentError(
                trainer.testOnClassData(dataset=testing_dataset),
                testing_dataset['class'])

            if tstresult > last_overloaded:
                break

            last_overloaded = tstresult

            if tstresult <= 0.001:
                if not os.path.exists(os.path.join(NN_PATH, 'cgm')):
                    os.makedirs(os.path.join(NN_PATH, 'cgm'))

                result = "epoch: %4d  train err: %5.2f%%  test err: %5.2f%%\n" \
                         % (trainer.totalepochs, trnresult, tstresult)

                result_file = open(os.path.join(NN, 'cgm',
                    '%s_%s' % (self.hidden_neuron, trainer.totalepochs)), 'w')
                result_file.write(result)
                result_file.close()

                network_path = os.path.join(NN_PATH, 'cgm',
                    '%s_%s.xml' % (self.hidden_neuron, trainer.totalepochs))
                NetworkWriter.writeToFile(self.cgm, network_path)

