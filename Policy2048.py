import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

from MockModel import MockModel

GAMMA = 0.99
CLIP_RATIO = 0.2
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def create_2048_policy_model():
    # Input layer
    mask_input = Input(shape=(4,), dtype='bool', name="mask_input")
    board_input = Input(shape=(4, 4), name="board_input")  # 1 channel for the board state

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(board_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Flatten
    x = Flatten()(x)

    # Dense layers
    x = Dense(128, activation='relu')(x)

    # Two heads
    # Value head
    value = Dense(1, activation='tanh', name="value_head")(x)

    def masked_softmax(logits, mask, axis=-1):
        # Subtract large negative value from positions of illegal moves
        # before applying softmax, which makes their probability nearly zero.
        masked_logits = tf.where(mask, logits, logits - 1e9)
        return tf.nn.softmax(masked_logits, axis=axis)

    # Apply the mask on the policy logits before softmax
    policy_logits = Dense(4, name="policy_logits")(x)
    policy = tf.keras.layers.Lambda(lambda x: masked_softmax(x[0], x[1]), name="policy_head")([policy_logits, mask_input])

    # Construct the model
    model = Model(inputs=[board_input, mask_input], outputs=[value, policy])
    model.compile(optimizer='adam',
                  loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
                  metrics={'policy_head': 'accuracy'})

    return model

def create_non_masked_2048_policy_model():
    # Input layer
    board_input = Input(shape=(4, 4), name="board_input")  # 1 channel for the board state

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(board_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Flatten
    x = Flatten()(x)

    # Dense layers
    x = Dense(128, activation='relu')(x)

    # Two heads
    # Value head
    value = Dense(1, activation='tanh', name="value_head")(x)

    def masked_softmax(logits, mask, axis=-1):
        # Subtract large negative value from positions of illegal moves
        # before applying softmax, which makes their probability nearly zero.
        masked_logits = tf.where(mask, logits, logits - 1e9)
        return tf.nn.softmax(masked_logits, axis=axis)

    # Apply the mask on the policy logits before softmax
    policy = Dense(4, name="policy_head", activation="softmax")(x)

    # Construct the model
    model = Model(inputs=[board_input], outputs=[value, policy])
    model.compile(optimizer='adam',
                  loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
                  metrics={'policy_head': 'accuracy'})

    return model

def create_mock_2048_policy_model():
    return MockModel()