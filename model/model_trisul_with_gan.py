
import tensorflow as tf
import keras
from spektral.layers import GCNConv
import numpy as np
import networkx as nx
from tensorflow.keras.layers import Input,Dropout
from tensorflow.keras.regularizers import l2


def make_model(m_size, u_size, m_rate_size, u_rate_size,window_size):
    def concat(x):
        w_size, d_size= x.shape
        x_res = tf.tile(x, multiples=[1, w_size])
        x_res= tf.reshape(x_res, (-1,d_size,))

        y_res= tf.tile(x, multiples=[w_size, 1])
        res = tf.concat([x_res, y_res], axis=1)
        res= tf.reshape(res, (-1,w_size, d_size+d_size))
        return res

    def mat_mul_k(x):
        return tf.keras.backend.dot(x[0], x[1])

    def GCN_Layer(ip_dim, op_dim, priv_layer):
        concated= tf.keras.layers.Lambda(concat, dynamic=True, output_shape=(window_size, ip_dim+ip_dim,))(priv_layer)
#         print(concated.shape)
        d1= tf.keras.layers.Dense(units= 1)(concated)
#         print(d1.shape)
        d1_re= tf.keras.layers.Reshape((window_size,))(d1)
#         print(d1_re.shape)
        le_re= tf.keras.layers.LeakyReLU()(d1_re)
#         print(le_re.shape)
        do_l= tf.keras.layers.Dropout(0.2)(le_re)
#         print(do_l.shape)
        a1= tf.keras.layers.Softmax()(do_l)
#         print(a1.shape)
        a_cross_x= tf.keras.layers.Lambda(mat_mul_k, dynamic=False)([a1,priv_layer])
#         print(a_cross_x.shape)
        op= tf.keras.layers.Dense(units= op_dim, activation= "sigmoid")(a_cross_x)
#         print(op.shape)
        return op
    
    def gcn_parallel_layers(ip_dim, op_dim, priv_layer, K_val=8):
        if K_val==1:
            return GCN_Layer(ip_dim, op_dim, priv_layer)
        layers= []
        for k_id in range(K_val):
            l_= GCN_Layer(ip_dim, op_dim, priv_layer)
            layers.append(l_)
        avg= tf.keras.layers.Average()(layers)
        return avg
    ip_m= keras.layers.Input(shape=(m_size,), name="input_m")
    ip_u= keras.layers.Input(shape=(u_size,), name="input_u")
    d_r1_m= keras.layers.Dense(units= int(m_size*2/3), activation="tanh")(ip_m)
    d_r1_u= keras.layers.Dense(units= int(m_size*2/3), activation="tanh")(ip_u)

    d_r_op_m= keras.layers.Dense(units= m_rate_size, activation="sigmoid", name="out_movie_rate")(d_r1_m)
    d_r_op_u= keras.layers.Dense(units= u_rate_size, activation="sigmoid", name="out_user_rate")(d_r1_u)

    d_m= keras.layers.Dense(units= int(m_size*2/3), activation="tanh")(d_r_op_m)
    d_u= keras.layers.Dense(units= int(u_size*2/3), activation="tanh")(d_r_op_u)
    gcn_layer= gcn_parallel_layers(m_size, int(m_size*2/3), ip_m, K_val=8)
    elu= tf.keras.layers.ELU(alpha= 0.2)(gcn_layer)
    concated= keras.layers.concatenate([d_m, d_u,elu])
    d_c1= keras.layers.Dense(units= int((u_size+m_size)/2), activation="tanh")(concated)
    d_c2= keras.layers.Dense(units=1024, activation="sigmoid")(d_c1)
    d_c3= keras.layers.Dense(units=128, activation="sigmoid")(d_c2)
    d_c4= keras.layers.Dense(units=1, activation="sigmoid", name= "out_norm_rate")(d_c3)

    model= keras.models.Model([ip_m, ip_u], [d_c4, d_r_op_m, d_r_op_u])
    model.compile(loss="MSE", optimizer="adam")

    return model





if __name__ == "__main__":
    m_size, u_size= 1220, 1266
    m_rate_size, u_rate_size= 900, 1600
    batch_size=64
    m= make_model(m_size, u_size, m_rate_size, u_rate_size,window_size= batch_size)
    m.save("./models/model_trisul.h5")
