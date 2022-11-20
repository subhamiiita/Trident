import helpers
import random
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import mean_squared_error as mse

import model_trisul_with_gan as neural_network





# path to specify movie and user representation

movie_enc= helpers.load_pkl("../objs/enc_movie_with_meta+video.obj")
user_enc= helpers.load_pkl("../objs/enc_user.obj")


save_model_name= "trident.h5"

#path to specify movie and user rating vector representataion

movie_enc_rates= helpers.load_pkl("../objs/enc_movie_from_rates.obj")
user_enc_rates= helpers.load_pkl("../objs/enc_user_from_rates.obj")


#path to specify training and testing sample
    
train_obj= helpers.load_pkl("../objs/u1.train.obj")
test_obj= helpers.load_pkl("../objs/u1.test.obj")


#to get the shape of different embeddings

movie_shape= len(movie_enc.get(list(movie_enc.keys())[0]))
user_shape= len(user_enc.get(list(user_enc.keys())[0]))



movie_rate_shape= len(movie_enc_rates.get(list(movie_enc_rates.keys())[0]))
user_rate_shape= len(user_enc_rates.get(list(user_enc_rates.keys())[0]))


#define batch size

batch_size= 64




def shuffle_single_epoch(ratings):
    data_copied= ratings.copy()
    random.shuffle(data_copied)
    return data_copied


def normalize(rate):
    return rate/5


def de_normalize(rate):
    return rate*5


#this is done to keep training and testing samples in the multiple of batch size

def copy_to_fix_shape(movie, user, rate, movie_from_rate, user_from_rate, req_len):
    assert len(movie)==len(user)==len(rate)
    new_movie, new_user, new_rate, new_movie_from_rate, new_user_from_rate= movie[:], user[:], rate[:], movie_from_rate[:], user_from_rate[:]
    while len(new_movie)<req_len:
        r_ind= random.randint(0, len(movie)-1)
        new_movie.append(movie[r_ind])
        new_user.append(user[r_ind])
        new_rate.append(rate[r_ind])
        new_movie_from_rate.append(movie_from_rate[r_ind])
        new_user_from_rate.append(user_from_rate[r_ind])
    return new_movie, new_user, new_rate, new_movie_from_rate, new_user_from_rate


#code to get a particular batch

def get_nth_batch(ratings, n, batch_size= batch_size, take_entire_data= False):
    users= []
    movies= []
    rates= []
    user_from_rate= []
    movie_from_rate= []
    if take_entire_data:
        slice_start= 0
        slice_end= len(ratings)
    else:
        if (n+1)*batch_size>len(ratings):
            print("OUT OF RANGE BATCH ID")
        slice_start= n*batch_size
        slice_end= (n+1)*batch_size
    for user_id, movie_id, rate in ratings[slice_start: slice_end]:
        if user_enc.get(user_id) is None or movie_enc.get(movie_id) is None:
            continue
        users.append(user_enc.get(user_id))
        movies.append(movie_enc.get(movie_id))
        rates.append(normalize(rate))

        user_from_rate.append(user_enc_rates.get(user_id))
        movie_from_rate.append(movie_enc_rates.get(movie_id))
    if not take_entire_data:
        movies, users, rates, movie_from_rate, user_from_rate= copy_to_fix_shape(movies, users, rates, movie_from_rate, user_from_rate, batch_size)
    users= np.array(users)
    movies= np.array(movies)
    rates= np.array(rates)
    user_from_rate= np.array(user_from_rate)
    movie_from_rate= np.array(movie_from_rate)


    return movies, users, rates, movie_from_rate, user_from_rate

#code to train

def train(model, data, test_data= None, no_of_epoch= 32,recurs_call=False):
    total_batches_train= int(len(data)/batch_size)
    for epoch in range(no_of_epoch):
        print("\n\n---- EPOCH: ", epoch, "------\n\n")
        data= shuffle_single_epoch(data)
        for batch_id in range(total_batches_train):
            print("Epoch: ", epoch+1, " Batch: ", batch_id)
            movies, users, rates, movie_from_rate, user_from_rate= get_nth_batch(data, batch_id)
            if len(rates)==0:
#                 print("not found batch_id = ",batch_id)
                continue
            model.fit([movies, users], [rates, movie_from_rate, user_from_rate], batch_size=batch_size, epochs=1, verbose=2)
        if test_data is not None:
            test(model, test_data, take_entire_data=False, save=True)
      
    
    while True and not recurs_call:
        some_last_rmse= all_rmse[-7:]
        best_rmse= min(all_rmse)
        if best_rmse not in some_last_rmse:
            break
        best_in_last= False
        for l in some_last_rmse:
            if abs(l-best_rmse)<0.01:
                best_in_last= True
        if not best_in_last: break

        train(model, data, test_data=test_data, no_of_epoch=1, recurs_call=True)



lest_rmse= float("inf")
all_rmse= list()
def test(model, data, save= True, take_entire_data= True):
    if take_entire_data:
        movies, users, res_true, _, _= get_nth_batch(data, 0, take_entire_data=take_entire_data)
        res_pred,_,_= model.predict([movies, users], batch_size=batch_size)
        res_true= np.array(res_true)
        res_pred= np.array(res_pred).reshape(-1)
        assert len(res_true)==len(res_pred)
    else:
        total_batches_test=int(len(data)/batch_size)
        res_true, res_pred= np.array([]), np.array([])
        print("total_batches_test = ",total_batches_test)
        for batch_id in range(total_batches_test+1):
            movies, users, rates, _, _= get_nth_batch(data, batch_id)
            if len(rates)==0:
#                 print("not found batch_id = ",batch_id)
                continue
            pred, _, _= model.predict([movies, users], batch_size=batch_size)
            pred= np.array(pred).reshape(-1)
            assert len(rates)==len(pred)
            res_true= np.concatenate([res_true, rates])
            res_pred= np.concatenate([res_pred, pred])
    y_true= de_normalize(res_true)
    y_pred= de_normalize(res_pred)


#calculate evaluation metric

    rmse= calc_rms(y_true, y_pred)
    y_pred= np.array([round(x) for x in y_pred])
    rmse_n= calc_rms(y_true, y_pred)
    all_rmse.append(rmse)
    print("rmse: ", rmse, " rmse_norm: ", rmse_n)
    global lest_rmse
    if save and lest_rmse>rmse:
        lest_rmse= rmse
        helpers.save_model(model, save_filename=save_model_name)

def calc_rms(t, p):
    return mse(t, p, squared=False)


def train_test_ext(train_obj, test_obj):
    model= neural_network.make_model(movie_shape, user_shape, movie_rate_shape, user_rate_shape,window_size=batch_size)
    train(model, data=train_obj, test_data=test_obj)
    test(model, data= test_obj, save= False, take_entire_data=False)


def test_saved(saved_model, test_file_path):
    import keras
    model= keras.models.load_model(saved_model)
    ratings_test_path= helpers.path_of(test_file_path)
    test_obj= helpers.load_pkl(ratings_test_path)
    test(model, data= test_obj, take_entire_data=True)


#main function defined

if __name__=="__main__":

    if len(sys.argv) in [2,3]:
        model_loc= sys.argv[1]
        if model_loc.split(".")[-1]!="h5":
            print("Tried to load model that is not h5 format")
            exit()
        if len(sys.argv)==3:
            test_file_loc= sys.argv[2]
            if test_file_loc.split(".")[-1]!="obj":
                print("Test file must be in .obj format")
        else:
            test_file_loc= "../liv_data/objs/splits/u1.test.obj"

        test_saved(model_loc, test_file_loc)
    
    else:



        

        print("Movie_shape: ", movie_shape)
        print("User_shape: ", user_shape)

        print("\nMovie_rate_shape: ", movie_rate_shape)
        print("User_rate_shape: ", user_rate_shape)

        print("batch_size:", batch_size)

        print("Saved_model_name: ", save_model_name)

        print("\n\n")

        train_test_ext(train_obj, test_obj)
