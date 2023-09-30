import tensorflow as tf
import matplotlib.pyplot as plt
import models.resnet as resnet


def view(anchors, positives, negatives):
    n = anchors.shape[0]
    _, ax = plt.subplots(n, 3)
    print(negatives.shape)
    for i, image in enumerate(anchors):
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(positives[i])
        ax[i, 2].imshow(negatives[i])
    plt.waitforbuttonpress(1)


class Siamese(tf.keras.Model):
    def __init__(self, config_model_dict: dict[str, str], config_data_dict: dict[str, str], **kwargs):
        super(Siamese, self).__init__(**kwargs)
        self.config_model_dict = config_model_dict
        self.config_data_dict = config_data_dict
        self.CROP_SIZE = int(config_data_dict["CROP_SIZE"])
        self.PROJECT_DIM = int(config_model_dict["PROJECT_DIM"])
        self.WEIGHT_DECAY = float(config_model_dict["WEIGHT_DECAY"])
        self.MARGIN = float(config_model_dict["MARGIN"])
        self.CHANNELS = 3
        self.encoder = self.get_encoder()  # split into sketch and photo encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dist_pos_tracker = tf.keras.metrics.Mean(name="dist_pos")
        self.dist_neg_tracker = tf.keras.metrics.Mean(name="dist_neg")

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "config_model_dict": self.config_model_dict,
                "config_data_dict": self.config_data_dict,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_encoder(self):
        inputs = tf.keras.layers.Input((self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS))
        x = inputs  # / 127.5 - 1
        # the backbone can be an input to the clas SimSiam
        bkbone = resnet.ResNetBackbone(
            [3, 4, 6, 3],
            [64, 128, 256, 512],
            kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY),
        )
        x = bkbone(x)
        # Projection head.
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.PROJECT_DIM)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(self.PROJECT_DIM)(x)

        #         #x = tf.keras.layers.Flatten()(x)
        #         x = tf.keras.layers.Dense(
        #             self.PROJECT_DIM,
        #             use_bias=True,
        #             kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY)
        #         )(x)
        # outputs = tf.keras.layers.BatchNormalization()(x)

        outputs = tf.math.l2_normalize(x, axis=1)

        return tf.keras.Model(inputs, outputs, name="encoder")

    def compute_loss(self, xa, xp, xn):
        margin = self.MARGIN
        dist_pos = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xp), axis=1))
        dist_neg = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xn), axis=1))
        # dist_pos  = tf.math.sqrt(2.0 - 2.0*tf.reduce_sum((xa * xp), axis = 1))
        # dist_neg  = tf.math.sqrt(2.0 - 2.0*tf.reduce_sum((xa * xn), axis = 1))
        loss = tf.math.maximum(0.0, dist_pos - dist_neg + margin)

        return tf.reduce_mean(loss), tf.reduce_mean(dist_pos), tf.reduce_mean(dist_neg)

    def train_step(self, batch):
        # Unpack the data.
        anchors, positives, negatives = batch
        # select negatives
        # n = tf.shape(anchors)[0]
        # pos = tf.range(n)
        # perm = tf.random.shuffle(pos)
        # perm = tf.where(perm == pos, (perm + 1) % n, perm)
        # negatives = tf.gather(anchors, perm)
        # view(anchors, positives, negatives)
        # training one step
        with tf.GradientTape() as tape:
            xa = self.encoder(anchors)
            xp = self.encoder(positives)
            xn = self.encoder(negatives)
            loss, dist_pos, dist_neg = self.compute_loss(xa, xp, xn)

        # Compute gradients and update the parameters.
        learnable_params = self.encoder.trainable_variables
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # tracking status.
        self.loss_tracker.update_state(loss)
        self.dist_pos_tracker.update_state(dist_pos)
        self.dist_neg_tracker.update_state(dist_neg)

        return {
            "loss": self.loss_tracker.result(),
            "dist_pos": self.dist_pos_tracker.result(),
            "dist_neg": self.dist_neg_tracker.result(),
        }

    def build(self):
        anchor_input = tf.keras.layers.Input(shape=(self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS), name="anchor_input")
        positive_input = tf.keras.layers.Input(shape=(self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS), name="positive_input")
        negative_input = tf.keras.layers.Input(shape=(self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS), name="negative_input")

        anchor_embedding = self.encoder(anchor_input)
        positive_embedding = self.encoder(positive_input)
        negative_embedding = self.encoder(negative_input)

        loss, _, _ = self.compute_loss(anchor_embedding, positive_embedding, negative_embedding)

        super(Siamese, self).__init__(inputs=[anchor_input, positive_input, negative_input], outputs=loss)
