import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_KERAS'] = '1'
from keras_transformers.backend import keras, K
from keras_transformers.models import build_bert_model
from keras_transformers.tokenizer import Tokenizer
from keras_transformers.utils import pad_sequences, DataGenerator, viterbi_decode
from keras_transformers.optimizers import Adam
from keras_transformers.loss import Loss
from keras_transformers.backend import softmax
from keras.layers import *
from keras.models import Model, Sequential

from tqdm import tqdm
import numpy as np
import json

from keras_transformers.optimizers import (
    wrap_optimizer_with_warmup,
    wrap_optimizer_with_weight_decay
)

import tensorflow as tf
from utils import  loss_kl, euclidean_metric

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=5e-5, help='learning rate')
parser.add_argument("--bs", type=int, default=32, help='batch size')
parser.add_argument("--q", type=float, default=0.1, help='q')
parser.add_argument("--tau", type=float, default=0.05, help='tau')
parser.add_argument("--seed", type=int, default=1, help="seed")
parser.add_argument("--group", type=str, default="intra", help="signature")
parser.add_argument("--way", type=int, default=5, help="n way")
parser.add_argument("--shot", type=int, default=1, help="k shot")
parser.add_argument("--index", type=int, default=0, help="file index")
parser.add_argument("--alpha", type=float, default=0.7, help="file index")
parser.add_argument("--temperature", type=float, default=1., help="alpha")

args = parser.parse_args()
alpha = args.alpha
temperature = args.temperature
scale_by_temperature = True


def read_data(path):
    D = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            D.append(json.loads(line))
    return D


with open(f'datasets/Ontonotes5/label2nl.json', 'r', encoding='utf-8') as fr:
    entity_label = json.load(fr)

#with open('datasets/FewNerd/word_mapping.jsonl', 'r', encoding='utf-8') as fr:
#    word_mapping = json.load(fr)


def setup_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def get_labels(text, labels):
    words_list = text.split()
    label_list = [0] * len(words_list)
    spans = [(len(text[:s].strip().split()), len(text[s:e + 1].split()), categories.index(l) + 1)
             for s, e, l in labels]
    for s, t, l in spans:
        label_list[s:s+t] = [l] * t
    return zip(words_list, label_list)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        for data in self.batch_generator:
            batch_token_ids, batch_segment_ids, batch_labels, batch_token_masks, batch_cls_ids = [], [], [], [], []
            label_token_ids, _ = tokenizer.encode(categories_nl)
            label_token_ids = [token_id for token_ids in label_token_ids for token_id in token_ids]
            for d in data:
                words_labels = get_labels(d[0], d[1:])
                token_list = [tokenizer.TOKEN_CLS]
                labels = [-100]
                for word, label in words_labels:
                    tokens = tokenizer.tokenize(word)[1:-1]
                    token_list += tokens
                    labels += [label] + [-100] * (len(tokens) - 1)
                token_list.append(tokenizer.TOKEN_SEP)
                labels.append(-100)
                while len(token_list) > maxlen:
                    token_list.pop(-2)
                    labels.pop(-2)
                
                token_masks = [1] * len(token_list)
                token_ids = tokenizer.convert_tokens_to_ids(token_list)
                token_ids += label_token_ids
                segment_ids = [0] * len(token_ids)
                labels += [-100] * (len(token_ids) - len(labels))
                token_masks += [0] * (len(token_ids) - len(token_masks))
                cls_ids = [i for i, v in enumerate(token_ids) if v == 101][1:]

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                batch_token_masks.append(token_masks)
                batch_cls_ids.append(cls_ids)

            batch_token_ids = pad_sequences(batch_token_ids)
            batch_segment_ids = pad_sequences(batch_segment_ids)
            batch_labels = pad_sequences(batch_labels, value=-100)
            batch_token_masks = pad_sequences(batch_token_masks)
            batch_cls_ids = pad_sequences(batch_cls_ids)

            yield [batch_token_ids, batch_segment_ids, batch_labels, batch_token_masks, batch_cls_ids], None


class ContrastiveLoss(Loss):
    #def compute_loss(self, inputs, mask=None):
    #    y_preds, labels, masks, original_embedding_mu, original_embedding_sigma = inputs
    #    masks = tf.reshape(masks, [-1])
    #    labels = tf.reshape(labels, [-1])
    #    y_preds = tf.reshape(y_preds, [-1, y_preds.shape[-1]])
    #    original_embedding_mu = tf.reshape(original_embedding_mu, [-1, original_embedding_mu.shape[-1]])
    #    original_embedding_sigma = tf.reshape(original_embedding_sigma, [-1, original_embedding_sigma.shape[-1]])

    #    active_indices = tf.where(masks == 1)
    #    labels = tf.gather_nd(labels, active_indices)
    #    y_preds = tf.gather_nd(y_preds, active_indices)
    #    original_embedding_mu = tf.gather_nd(original_embedding_mu, active_indices)
    #    original_embedding_sigma = tf.gather_nd(original_embedding_sigma, active_indices)

    #    real_label_indices = tf.where(labels >= 0)
    #    labels = tf.gather_nd(labels, real_label_indices)
    #    y_preds = tf.gather_nd(y_preds, real_label_indices)
    #    original_embedding_mu = tf.gather_nd(original_embedding_mu, real_label_indices)
    #    original_embedding_sigma = tf.gather_nd(original_embedding_sigma, real_label_indices)

    #    active_nums = len(labels)
    #    embedding_dims = original_embedding_mu.shape[-1]
    #    repeated_embedding_mu = tf.repeat(original_embedding_mu, active_nums, axis=0)
    #    repeated_embedding_sigma = tf.repeat(original_embedding_sigma, active_nums, axis=0)
    #    tiled_embedding_mu = tf.tile(original_embedding_mu, [active_nums, 1])
    #    tiled_embedding_sigma = tf.tile(original_embedding_sigma, [active_nums, 1])
    #    similarity = loss_kl(repeated_embedding_mu, repeated_embedding_sigma, tiled_embedding_mu, tiled_embedding_sigma, embedding_dims)
    #    similarity = tf.reshape(similarity, (active_nums, active_nums))

    #    positive_masks = tf.cast(labels[:, None] == labels[None, :], tf.float32)
    #    negative_masks = 1 - tf.eye(tf.shape(similarity)[0])
    #    positive_masks = positive_masks * negative_masks

    #    loss = tf.exp(similarity)
    #    positive_nums = tf.reduce_sum(positive_masks, -1)
    #    positive_scores = tf.reduce_sum(loss * positive_masks, -1)
    #    negative_scores = tf.reduce_sum(loss * negative_masks, -1)

    #    non_zero_indices = tf.where(positive_nums > 0)
    #    positive_nums = tf.gather_nd(positive_nums, non_zero_indices)
    #    positive_scores = tf.gather_nd(positive_scores, non_zero_indices)
    #    negative_scores = tf.gather_nd(negative_scores, non_zero_indices)

    #    loss = -tf.math.log(positive_scores) + tf.math.log(negative_scores) + tf.math.log(positive_nums)
    #    loss = loss / tf.math.log(2.)
    #    return tf.reduce_mean(loss)

    def compute_loss(self, inputs, mask=None):
        """ Sum outside
        """
        y_preds, labels, masks, original_embedding_mu, original_embedding_sigma, cls_ids = inputs
        masks = tf.reshape(masks, [-1])
        labels = tf.reshape(labels, [-1])
        y_preds = tf.reshape(y_preds, [-1, y_preds.shape[-1]])
        original_embedding_mu = tf.reshape(original_embedding_mu, [-1, original_embedding_mu.shape[-1]])
        original_embedding_sigma = tf.reshape(original_embedding_sigma, [-1, original_embedding_sigma.shape[-1]])

        active_indices = tf.where(masks == 1)
        labels = tf.gather_nd(labels, active_indices)
        y_preds = tf.gather_nd(y_preds, active_indices)
        original_embedding_mu = tf.gather_nd(original_embedding_mu, active_indices)
        original_embedding_sigma = tf.gather_nd(original_embedding_sigma, active_indices)

        real_label_indices = tf.where(labels >= 0)
        labels = tf.gather_nd(labels, real_label_indices)
        y_preds = tf.gather_nd(y_preds, real_label_indices)
        original_embedding_mu = tf.gather_nd(original_embedding_mu, real_label_indices)
        original_embedding_sigma = tf.gather_nd(original_embedding_sigma, real_label_indices)

        active_nums = len(labels)
        embedding_dims = original_embedding_mu.shape[-1]
        repeated_embedding_mu = tf.repeat(original_embedding_mu, active_nums, axis=0)
        repeated_embedding_sigma = tf.repeat(original_embedding_sigma, active_nums, axis=0)
        tiled_embedding_mu = tf.tile(original_embedding_mu, [active_nums, 1])
        tiled_embedding_sigma = tf.tile(original_embedding_sigma, [active_nums, 1])
        logits = loss_kl(repeated_embedding_mu, repeated_embedding_sigma, tiled_embedding_mu, tiled_embedding_sigma, embedding_dims)
        logits = tf.reshape(logits, (active_nums, active_nums))

        positive_masks = tf.cast(labels[:, None] == labels[None, :], tf.float32)
        negative_masks = 1 - tf.eye(tf.shape(logits)[0])
        positive_masks = positive_masks * negative_masks
        positive_nums = tf.reduce_sum(positive_masks, -1)

        logits = logits / temperature
        logits = logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
        exp_logits = tf.exp(logits)

        negative_scores = tf.reduce_sum(exp_logits * negative_masks, axis=1, keepdims=True)
        log_probs = (logits - tf.math.log(negative_scores)) * positive_masks
        log_probs = tf.reduce_sum(log_probs, axis=1)
        log_probs = tf.math.divide_no_nan(log_probs, positive_nums)

        loss = -log_probs
        if scale_by_temperature:
            loss *= temperature

        num_nonzeros = tf.math.count_nonzero(loss)
        num_nonzeros = tf.cast(num_nonzeros, tf.float32)
        inner_loss = tf.reduce_sum(loss) / num_nonzeros

        # cross loss
        #y_preds = inputs[0]
        #label_repr = tf.gather(y_preds, K.cast(cls_ids, 'int32'), batch_dims=1)
        #logits_cross = euclidean_metric(y_preds, label_repr)
        #logits_cross = logits_cross / 0.05
        #probs_cross = tf.nn.softmax(logits_cross, -1)
        #probs_cross = tf.reshape(probs_cross, [-1, tf.shape(probs_cross)[-1]])
        #probs_cross = tf.gather_nd(probs_cross, active_indices)
        #probs_cross = tf.gather_nd(probs_cross, real_label_indices)
        #loss_cross = K.sparse_categorical_crossentropy(labels, probs_cross)
        
        # cross loss with distribution
        original_embedding_mu, original_embedding_sigma, cls_ids = inputs[3:]
        label_mu = tf.gather(original_embedding_mu, K.cast(cls_ids, 'int32'), batch_dims=1)
        label_sigma = tf.gather(original_embedding_sigma, K.cast(cls_ids, 'int32'), batch_dims=1)

        repeated_embedding_mu = tf.repeat(original_embedding_mu[:, :, None], tf.shape(cls_ids)[-1], axis=2)
        repeated_embedding_sigma = tf.repeat(original_embedding_sigma[:, :, None], tf.shape(cls_ids)[-1], axis=2)

        tile_label_mu = tf.repeat(label_mu[:, None], tf.shape(original_embedding_mu)[1], axis=1)
        tile_label_sigma = tf.repeat(label_sigma[:, None], tf.shape(original_embedding_mu)[1], axis=1)

        repeated_embedding_mu = tf.reshape(repeated_embedding_mu, [-1, tf.shape(repeated_embedding_mu)[-1]])
        repeated_embedding_sigma = tf.reshape(repeated_embedding_sigma, [-1, tf.shape(repeated_embedding_sigma)[-1]])
        tile_label_mu = tf.reshape(tile_label_mu, [-1, tf.shape(tile_label_mu)[-1]])
        tile_label_sigma = tf.reshape(tile_label_sigma, [-1, tf.shape(tile_label_sigma)[-1]])
        
        cross_logits = loss_kl(repeated_embedding_mu, repeated_embedding_sigma, tile_label_mu, tile_label_sigma, embedding_dims)
        cross_logits = tf.reshape(cross_logits, [tf.shape(original_embedding_mu)[0], tf.shape(original_embedding_mu)[1], tf.shape(cls_ids)[-1]])
        cross_probs = tf.nn.softmax(cross_logits, -1)

        cross_probs = tf.reshape(cross_probs, [-1, tf.shape(cross_probs)[-1]])  # len(categories)+1
        cross_probs = tf.gather_nd(cross_probs, active_indices)
        cross_probs = tf.gather_nd(cross_probs, real_label_indices)
        cross_loss = K.sparse_categorical_crossentropy(labels, cross_probs)

        return 0.1 * inner_loss + 0.9 * tf.reduce_mean(cross_loss)



    def compute_metrics(self, inputs, mask=None):
        y_pred, y_true, mask, _, _, _ = inputs
        mask = K.cast(mask, K.floatx())
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        accuracy = K.cast(K.equal(y_true, y_pred), K.floatx())
        return K.sum(accuracy * mask) / K.sum(mask)



# 数据加载
categories_nl = ['none']
for entity in entity_label:
    categories_nl.append(entity_label[entity])

categories = list(entity_label.keys())
print('categories:', categories)
print('categories_nl:', categories_nl)

train_data = read_data(f'datasets/Ontonotes5/example.train')
train_generator = data_generator(train_data, batch_size=args.bs)


# 超参数设置
maxlen = 128
epochs = 1
seed = args.seed

print('hyper parameters:\n', json.dumps({
    "batch_size": args.bs,
    "learning rate": args.lr,
}))

# 设置随机种子
setup_seed(seed)

# bert配置
config_path = './uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './uncased_L-12_H-768_A-12/vocab.txt'


ontonotes_save_path = f'./Ontonotes5/onto-project_seed_{args.seed}_uncased_out_prompt_cross_dist_alpha_0.1'
if not os.path.exists(ontonotes_save_path):
    os.makedirs(ontonotes_save_path)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 构建model
Encoder = build_bert_model(
    config_path,
    checkpoint_path,
)


labels = Input(shape=(None,))
token_mask = Input(shape=(None,))
cls_ids = Input(shape=(None,))


token_output = Encoder.output

embedder_mu = Sequential([
    ReLU(),
    Dense(128)
])
embedder_sigma = Sequential([
    ReLU(),
    Dense(128)
])


outputs = Dropout(0.1)(token_output)
original_embedding_mu = embedder_mu(outputs)
original_embedding_sigma = Lambda(lambda x: x[0] + x[1])([tf.keras.layers.ELU()(embedder_sigma(outputs)), 1 + 1e-14])


outputs = ContrastiveLoss(output_dims=0, metrics='sparse_accuracy')([token_output, labels, token_mask,
                                                                     original_embedding_mu, original_embedding_sigma,
                                                                     cls_ids])
model = Model(Encoder.inputs + [labels, token_mask, cls_ids], outputs)

adamWD = wrap_optimizer_with_weight_decay(Adam)
model.compile(
    optimizer=adamWD(
        learning_rate=args.lr,
        exclude_weights=['Norm', 'bias'],
    ),
)
model.summary()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):

        model.save_weights(save_path + '/best_model.weights')



if __name__ == '__main__':
    save_path = ontonotes_save_path
    test_data = None
    model.fit(
        train_generator.fit_generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[Evaluator()]
    )

