class Singleton(object):
    __instance=None
    def __init__(self):
        pass
    def getInstance(self):
        if Singleton.__instance is None:
            # Singleton.__instance=object.__new__(cls,*args,**kwd)
            Singleton.__instance=self.getTestFlag()
            print("build FLAGS over")
        return Singleton.__instance
    def get_andy_flag(self):
        import tensorflow as tf
        flags = tf.app.flags

 
        flags.DEFINE_string("dataset", "netflix_three_month", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("model_type", "joint", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("pretrained_model", "mf-25-0.12267.pkl", "Comma-separated list of hostname:port pairs")
        # flags.DEFINE_string("model_type", "mf", "Comma-separated list of hostname:port pairs")
        # flags.DEFINE_string("model_type", "joint", "Comma-separated list of hostname:port pairs")

        # flags.DEFINE_string("test_file_name", "ua.test", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("train_file_name", "ratings_subset.csv", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("work_dir", "online_model", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_integer("export_version", "80", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_integer("subset_size", 100, "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("moviesLen_100k_split_data", "1998-03-08", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("netflix_6_mouth_split_data", "2005-12-01", "Comma-separated list of hostname:port pairs")

        flags.DEFINE_integer("batch_size", 128, "Batch size of data while training")
        flags.DEFINE_integer("gan_k", 128, "Batch size of data while training")
            
        flags.DEFINE_integer("user_delta", 7, "Batch size of data while training")
        flags.DEFINE_integer("item_delta", 7, "Batch size of data while training")  # TODO :  user_delta could not equals to item_delta
        flags.DEFINE_integer("re_rank_list_length", 25, "Batch size of data while training")
        flags.DEFINE_integer("item_windows_size", 4, "Batch size of data while training")
        flags.DEFINE_integer("user_windows_size", 4, "Batch size of data while training")
        flags.DEFINE_integer("n_epochs", 10, "Batch size of data while training")
        flags.DEFINE_integer("test_granularity_count", 2, "Batch size of data while training")
        flags.DEFINE_integer("mf_embedding_dim", 10, "Batch size of data while training")
        flags.DEFINE_integer("rnn_embedding_dim", 10, "Batch size of data while training")  
        flags.DEFINE_integer("g_epoch_size", 2, "Batch size of data while training")  
        flags.DEFINE_integer("d_epoch_size", 1, "Batch size of data while training")  



        flags.DEFINE_float("learning_rate", 0.005, "Batch size of data while training")#0.0001
        flags.DEFINE_float("grad_clip", 0.1, "Batch size of data while training")
        flags.DEFINE_float("lamda", 0.05, "Batch size of data while training")
        flags.DEFINE_float("temperature", 5, "Batch size of data while training")

        flags.DEFINE_float("momentum", 1, "Batch size of data while training")

        flags.DEFINE_integer("threshold", 300, "Erase the users if the number of rating less than threshold")
        flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")
        flags.DEFINE_boolean("pretrained", False, "Test accuracy")
        flags.DEFINE_boolean("is_sparse", True, "Test accuracy")
        flags.DEFINE_boolean("rating_flag", False, "Test accuracy")
        flags.DEFINE_boolean("dns", False, "Test accuracy")
        flags.DEFINE_boolean("lastone", True, "Test accuracy")
        flags.DEFINE_boolean("sparse_tensor", False, "Test accuracy")
        flags.DEFINE_boolean("pairwise", False, "Test accuracy")
        FLAGS= flags.FLAGS


        netflix_month={"start":"2005-06-01",
                        "split":"2005-12-01",
                        "end"  :"2005-13-01"
                }
        netflix_year={"start":"2004-06-01",
                        "split":"2005-06-01",
                        "end"  :"2005-07-00"
                }
        netflix_full={"start":"1999-12-01",
                        "split":"2005-12-01",
                        "end"  :"2005-13-01"
                }
        netflix_three_month={"start":"2005-09-01",
                        "split":"2005-12-01",
                        "end"  :"2005-13-01"
                }
        movieslen100k={"start":"1000-12-01",
                        "split":"1998-03-08",
                        "end"  :"3005-13-01"
                }
        date_dict={"netflix_six_month":netflix_month,"netflix_year":netflix_year,"netflix_full":netflix_full,"movieslen100k":movieslen100k,"netflix_three_month":netflix_three_month,"netflix_three_month_40":netflix_three_month}
        # train_split_data={"moviesLen_100k": FLAGS.moviesLen_100k_split_data , "netflix_6_mouth": FLAGS.netflix_6_mouth_split_data }
        FLAGS.split_data=date_dict.get(flags.FLAGS.dataset,None)["split"]

        # flags.DEFINE_string("split_data", "ratings.csv", "Comma-separated list of hostname:port pairs")

        if FLAGS.dataset.startswith("movies"):
            FLAGS.threshold=0
        # # FLAGS.workernum=4
        return FLAGS
    def get_waby_flag(self):
        import tensorflow as tf
        flags = tf.app.flags

 
        flags.DEFINE_string("dataset", "movieslen100k", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("model_type", "joint", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("pretrained_model", "mf-25-0.12267.pkl", "Comma-separated list of hostname:port pairs")
        # flags.DEFINE_string("model_type", "mf", "Comma-separated list of hostname:port pairs")
        # flags.DEFINE_string("model_type", "joint", "Comma-separated list of hostname:port pairs")

        # flags.DEFINE_string("test_file_name", "ua.test", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("train_file_name", "ratings_subset.csv", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("work_dir", "online_model", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_integer("export_version", "80", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_integer("subset_size", 100, "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("moviesLen_100k_split_data", "1998-03-08", "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("netflix_6_mouth_split_data", "2005-12-01", "Comma-separated list of hostname:port pairs")

        flags.DEFINE_integer("batch_size", 128, "Batch size of data while training")
        flags.DEFINE_integer("gan_k", 128, "Batch size of data while training")
            
        flags.DEFINE_integer("user_delta", 7, "Batch size of data while training")
        flags.DEFINE_integer("item_delta", 7, "Batch size of data while training")  # TODO :  user_delta could not equals to item_delta
        flags.DEFINE_integer("re_rank_list_length", 25, "Batch size of data while training")
        flags.DEFINE_integer("item_windows_size", 4, "Batch size of data while training")
        flags.DEFINE_integer("user_windows_size", 4, "Batch size of data while training")
        flags.DEFINE_integer("n_epochs", 10, "Batch size of data while training")
        flags.DEFINE_integer("test_granularity_count", 2, "Batch size of data while training")
        flags.DEFINE_integer("mf_embedding_dim", 16, "Batch size of data while training")
        flags.DEFINE_integer("rnn_embedding_dim", 10, "Batch size of data while training")  
        flags.DEFINE_integer("g_epoch_size", 2, "Batch size of data while training")  
        flags.DEFINE_integer("d_epoch_size", 1, "Batch size of data while training")  



        flags.DEFINE_float("learning_rate", 0.005, "Batch size of data while training")#0.0001
        flags.DEFINE_float("grad_clip", 0.1, "Batch size of data while training")
        flags.DEFINE_float("lamda", 0.05, "Batch size of data while training")
        flags.DEFINE_float("temperature", 5, "Batch size of data while training")

        flags.DEFINE_float("momentum", 1, "Batch size of data while training")

        flags.DEFINE_integer("threshold", 300, "Erase the users if the number of rating less than threshold")
        flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")
        flags.DEFINE_boolean("pretrained", False, "Test accuracy")
        flags.DEFINE_boolean("is_sparse", True, "Test accuracy")
        flags.DEFINE_boolean("rating_flag", False, "Test accuracy")
        flags.DEFINE_boolean("dns", False, "Test accuracy")
        flags.DEFINE_boolean("lastone", True, "Test accuracy")
        flags.DEFINE_boolean("sparse_tensor", False, "Test accuracy")
        flags.DEFINE_boolean("pairwise", False, "Test accuracy")
        FLAGS= flags.FLAGS


        netflix_month={"start":"2005-06-00",
                        "split":"2005-12-00",
                        "end"  :"2005-13-00"
                }
        netflix_year={"start":"2004-06-00",
                        "split":"2005-06-00",
                        "end"  :"2005-07-00"
                }
        netflix_full={"start":"1999-12-00",
                        "split":"2005-12-00",
                        "end"  :"2005-13-00"
                }
        movieslen100k={"start":"1000-12-00",
                        "split":"1998-03-08",
                        "end"  :"3005-13-00"
                }
        date_dict={"netflix_six_month":netflix_month,"netflix_year":netflix_year,"netflix_full":netflix_full,"movieslen100k":movieslen100k}
        # train_split_data={"moviesLen_100k": FLAGS.moviesLen_100k_split_data , "netflix_6_mouth": FLAGS.netflix_6_mouth_split_data }
        FLAGS.split_data=date_dict.get(flags.FLAGS.dataset,None)["split"]

        # flags.DEFINE_string("split_data", "ratings.csv", "Comma-separated list of hostname:port pairs")

        if FLAGS.dataset.startswith("movies"):
            FLAGS.threshold=0
        # # FLAGS.workernum=4
        return FLAGS