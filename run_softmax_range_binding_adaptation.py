from mix_main_mul_dim_softmax_binding_adaptation import *
import json
import sys
import pickle
sys.path.append("/home/lucy/r10921071/GPdata/pmlb")

from pmlb import fetch_data

import mrmr


dataset = {
    "f2_gt":["strogatz_shearflow2",
         "strogatz_shearflow1"   ,
         "strogatz_predprey2"  ,
         "strogatz_predprey1"   ,
         "strogatz_lv2"   ,
         "strogatz_lv1"   ,
         "strogatz_glider2"  ,
         "strogatz_glider1",
         "strogatz_barmag2",
         "strogatz_barmag1",
         "strogatz_bacres2",
         "strogatz_bacres1",
    ],
    "f5":["690_visualizing_galaxy" ,
        "687_sleuth_ex1605", 
        "656_fri_c1_100_5",
        "649_fri_c0_500_5",
        "631_fri_c1_500_5",
        "628_fri_c3_1000_5",
        "624_fri_c0_100_5",
        "617_fri_c3_500_5",
        "613_fri_c3_250_5",
        "611_fri_c3_100_5"
    ],
    "f10":["666_rmftsa_ladata",
          "657_fri_c2_250_10"  ,
          "695_chatfield_4"  ,
          "654_fri_c0_500_10" ,
          "647_fri_c1_250_10" ,
          "646_fri_c3_500_10"],

    "mix":[
        "strogatz_shearflow2",
        "strogatz_shearflow1",
        "657_fri_c2_250_10"  ,
        "695_chatfield_4"  ,
        "690_visualizing_galaxy" ,
        "656_fri_c1_100_5",
    ],
     "all":["strogatz_shearflow2",
         "strogatz_shearflow1"   ,
         "strogatz_predprey2"  ,
         "strogatz_predprey1"   ,
         "strogatz_lv2"   ,
         "strogatz_lv1"   ,
         "strogatz_glider2"  ,
         "strogatz_glider1",
         "strogatz_barmag2",
         "strogatz_barmag1",
         "strogatz_bacres2",
         "strogatz_bacres1",
         "690_visualizing_galaxy",
        "687_sleuth_ex1605", 
        "656_fri_c1_100_5",
        "649_fri_c0_500_5",
        "631_fri_c1_500_5",
        "628_fri_c3_1000_5",
        "624_fri_c0_100_5",
        "617_fri_c3_500_5",
        "613_fri_c3_250_5",
        "611_fri_c3_100_5",
        "666_rmftsa_ladata",
          "657_fri_c2_250_10"  ,
          "695_chatfield_4"  ,
          "654_fri_c0_500_10" ,
          "647_fri_c1_250_10" ,
          "646_fri_c3_500_10"
    ],



    "more_f2":[
        "663_rabe_266",
        "712_chscase_geyser1",
        "banana",
        "feynman_I_29_4",
        "feynman_I_34_27",
        "feynman_I_39_1",
        "feynman_II_38_14"
    ]
}


function_set = [ CommonFunction["add"]  #0
                , CommonFunction["sub"] 
                ,  CommonFunction["mul"]
                ,  CommonFunction["div"] #3
                ,  CommonFunction["log"] #3
                ,  CommonFunction["sqrt"] #3
                ,  CommonFunction["pow2"] #3
                ,  CommonFunction["exp"] #3
                ,  CommonFunction["sin"]
                , CommonFunction["cos"]
                ]

GPGOMEA_data = ['airfoil' 
        , 'concrete_compress' 
        , 'energyCooling' 
        , 'energyHeating'  
        , 'tower'  
        , 'wineRed'
        , 'wineWhite'
        , 'yacht'] #

from numpy import genfromtxt #
data_path = "/home/hcliao/wzfang_refuge/new_dataset/all_data/"#

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("-g","--generation" , type=int)
parser.add_argument("-t","--time" , type=int )
parser.add_argument("-p","--population", type=int )
parser.add_argument("-d","--dataset")
parser.add_argument("-n","--file_name" , default="None")
args = parser.parse_args()

ij = 0
output = {}

datas = None

if args.dataset not in dataset:#
    datas = [args.dataset]
else:
    datas = dataset[args.dataset]
print("total length:", len(datas))

all_record = []
for data_ in datas:
    output[data_] = {}
    for br in [0.2]:
        fitness = []
        length = []
        print(str(ij) + "/" + str(len(datas)))
        ij+=1
        output[data_][br] = {}
        
        X, y = fetch_data(data_, return_X_y=True)
        most = []
        least =[]
        rejection = [0 for i in range(args.generation)]
        for t in range(args.time):  
            print( "--" + str(t) + "/" + str(args.time) + "--")
            test = Population(
                [ i for i in range(len(X[0])) ] , 
            function_set, 
            num_population= args.population , 
            generation= args.generation,
            data_X = X  ,
            data_y = y ,
            depth = 5, dataName = data_,binding_rate = br )

            
            
            test.run()
            file = open('test', 'wb')
            pickle.dump(test.bindingFitnessRecord, file)
            f = test.programs[0].fitness
            fitness.append(f)
            print(f)
            length.append(len(test.programs[0].program))
            
            for i in range(args.generation):
                rejection[i] += test.rejection[i] 

            #m_all = nlargest( 3 ,test.two_layer_counters[-1] , key=test.two_layer_counters[-1].get )
            # for g in range(args.generation):
            #     print(test.two_layer_counters[g][m_all[0]]/ sum( 
            #         [test.two_layer_counters[g][k] for k in test.two_layer_counters[g]]    ))
            # print("===")
            # for g in range(args.generation):
            #     print(test.two_layer_counters[g][m_all[1]]/ sum( 
            #         [test.two_layer_counters[g][k] for k in test.two_layer_counters[g]]    ))

            # print("===")
            # for g in range(args.generation):
            #     print(test.two_layer_counters[g][m_all[2]]/ sum( 
            #         [test.two_layer_counters[g][k] for k in test.two_layer_counters[g]]    ))

            # #print(test.rejection)
        for i in range(args.generation):
            rejection[i] = rejection[i]/args.time
        
        output[data_][br]["fitness"] = fitness
        #output[data_][br]["length"] = length
        #print(output[data_][br]["binding_rejection"])
        print(sum(fitness)/len(fitness))

        


        from datetime import datetime
        n = datetime.now()
        if args.file_name != "None":
            target_path = "_{}_{}__{}_{}_{}.json".format( br ,args.file_name , n.year , n.month , n.day)
            exp_store_path = "/home/hcliao/wzfang_refuge/GP_exp_result"
            with open( exp_store_path +  args.dataset + target_path, 'w') as f:
                json.dump(output, f)
        