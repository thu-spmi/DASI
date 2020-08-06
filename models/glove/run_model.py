'''
use the glove toolkit, change and run demo.sh
'''
import os
import stat
import pickle, json
import numpy as np


class Run_GloVe:
    def __init__(self, **kwargs):
        with open(kwargs['vocab_path']) as f:
            self.vocab_dict = json.load(f)
        self.training_data_filepath = '../../../'+kwargs['data_path']
        self.main_dir = os.path.abspath('.') # main dir of this whole project
        self.save_dir = kwargs['embedding_path']
        self.word_emb_size = kwargs['emb_dim']
        self.output_filename = kwargs['output_filename']

        self.glove_dir = os.path.join(
            os.path.join(os.path.join(self.main_dir, 'models'), 'glove'), 'glove') # glove dir

        with open(os.path.join(self.glove_dir, 'demo.sh'), 'r') as shell_file:
            self.lines = shell_file.read()

        self.lines = self.lines.replace(
            'CORPUS=training_data_filepath', 'CORPUS=%s'%self.training_data_filepath)
        self.lines = self.lines.replace(
            'SAVE_FILE=vectors', 'SAVE_FILE=%s'%self.output_filename)
        self.lines = self.lines.replace(
            'VECTOR_SIZE=vector_size', 'VECTOR_SIZE=%s'%self.word_emb_size)

        self.new_shell_filepath = os.path.join(
            self.glove_dir, 'run_%dd.sh' % self.word_emb_size)
        with open(self.new_shell_filepath, 'w') as new_shell_file:
            new_shell_file.write(self.lines)
        os.chmod(self.new_shell_filepath,stat.S_IRWXU)

    def train(self, model='glove'):
        os.chdir(self.glove_dir)
        os.system('make')
        os.system('./run_%dd.sh'%self.word_emb_size)

        # change vectors txt into dict pickle file, and save in save_dir
        emb_dict = {}

        with open(self.output_filename+'.txt') as f:
            for line in f:
                splits = line.split()
                word, emb = splits[0], splits[1:]
                if word in self.vocab_dict:
                    emb_dict[word] = np.fromstring(string='|'.join(emb),
                                                   dtype=np.float32,
                                                   sep='|')
        os.chdir(self.main_dir)
        with open(os.path.join(self.save_dir, self.output_filename+'.pkl'), 'wb') as f:
            pickle.dump(emb_dict, f)


