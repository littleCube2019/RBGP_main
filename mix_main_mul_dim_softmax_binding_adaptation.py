import random
from Function import CommonFunction
from Fitness import *
import copy 
from heapq import nlargest, nsmallest
from scipy.special import softmax
def cal_range( X ):
    return (  abs( max(X) - min(X) ) )

def get_hash(p , i):
    return p.program[p.parentIdx[i]]* 100 + p.program[i]*10 + p.nthChild[i]
class Population :
    def __init__ (self , 
                terminal_set , 
                function_set ,
                num_population , 
                generation , 
                data_X,
                data_y, 
                depth = 5 ,
                binding_rate = 0.1,
                 dataName = None ):
        self.binding_rate = binding_rate
        self.num_population = num_population 
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.dataName =   dataName
        self.dim = len(data_X[0])
        self.depth = depth
        self.generation = generation
        self.programs = [Program(terminal_set , function_set) for _ in range(self.num_population )  ]
        self.X = data_X
        self.y = data_y

        self.minX = [ min([x for x in data_X[:,i]])  for i in range(self.dim) ]
        self.maxX = [ max([x for x in data_X[:,i]])  for i in range(self.dim) ]
        
        self.rangeX = [ abs(self.maxX[i] - self.minX[i]) for i in range(len(self.maxX))  ]

        
        self.two_layer_counters = [0 for i in range(self.generation)]

        self.cluster_bigger = [[] , []] # funcs , args
        self.cluster_smaller = [[], []] # funcs , args
        
        self.target_range = abs( max(data_y) - min(data_y) )
        
        self.bigger_function = []
        self.smaller_function = []

        self.bindingFitnessRecord = []
        self.record = []
        
        for p in self.programs:
            p.fitness = get_fitness_MAD(p , data_X ,data_y)
        
        self.num_two_layer = len(self.function_set)
        self.chosen_k_largest = (self.num_two_layer * (self.num_two_layer - 1)) / 2   
        self.chosen_k_largest =  int(self.chosen_k_largest * self.binding_rate)
        
        #print(self.chosen_k_largest)
       
    def tournament_selection(self):
        a_idx = random.randint(0, len(self.programs)-1)
        b_idx = random.randint(0, len(self.programs)-1)
        a = self.programs[a_idx]
        b = self.programs[b_idx]
        if a.fitness < b.fitness:
            return a , a_idx
        else:
            return b , b_idx
  
    def one_point_crossover(self , donor , receiver  ):

        donor_cutpoint = random.randint(0,len(donor.program)-1)
        receiver_cutpoint = random.randint(0,len(receiver.program)-1)
        new_program =  donor.program[:donor_cutpoint] + receiver.program[receiver_cutpoint:  receiver.get_subtree(receiver_cutpoint) +1] + donor.program[donor.get_subtree(donor_cutpoint)+1:]

        return Program(self.terminal_set , self.function_set , program=new_program)
 
    def subtree_mutation(self , donor):
        new_chicken = Program(self.terminal_set  , self.function_set , self.depth)
        return self.one_point_crossover(donor , new_chicken)
    
   
    def run_once(self, index):
       
        self.binding_adaptation = [0,0,0]
        self.next_programs = [ self.programs[i] for i in range(self.num_population)]
        self.parent_func = [ self.programs[i] for i in range(self.num_population)]
        self.parent_args = [ self.programs[i] for i in range(self.num_population)]
        #self.rejection.append(0)
        
      
        for i in range(self.num_population):
            p = random.uniform(0, 1)
            if p < 0: # SGP crossover
                donor ,donor_index = self.tournament_selection()
                donor2 ,donor_index2 = self.tournament_selection()
                self.next_programs[i] = self.one_point_crossover(donor , donor2)
                self.next_programs[i].fitness = get_fitness_MAD(self.next_programs[i] , self.X , self.y)
            
            elif p < 1:
                    
                    donor ,donor_index = self.tournament_selection()
                    #donor_index = i
                    arg_receiver_idx = None
                    arg_receiver_func = None

                    start = donor.cut_point
                    end = donor.get_subtree(donor.cut_point)
                    
                    x = None
                    if donor.diff_range > 1: #bigger , need smaller
                        x = donor.diff_range-1
                        # more larger ，more positive
                    
                    
                    else:
                        x = 1/(donor.diff_range + 0.001)-1
                
               
                    # more smaller ，more negative
                    # temp large , range crossover more often
                    
                    
                
                    softmax_prob  = 1/(1 + np.exp(-(x * 0.01))) 
                    
                    if donor.diff_range > 1:  # target / ouput > 1 ==>  tar > out ==> out need bigger
                        coin = random.uniform(0.25,1)   
                        #coin = 0.6
                    else:
                        coin = random.uniform(0,0.75) #small
                      
                    #coin = random.uniform(0,1) #small
                    
                    t = len(self.terminal_set)
                    choice_1 = -1 * (donor.first_terminal+1)
                    choice_2 = -1 * random.randint(1,t)


                    # update for mutation
                    self.bigger_function = None
                    self.smaller_function = None
                    tries = 0
                    while self.bigger_function == None or self.smaller_function == None :
                        p = Program(function_set=self.function_set , terminal_set=self.terminal_set , depth=3)
                        for j in range(len(p.program)):
                            if p.program[j] < 0:
                                p.program[j] = -1 * (donor.first_terminal+1)
                        
                        p.execute(self.X)
                        
                        #p.arg_mean  = mean(p.values[0]) 
                        p.arg_range = cal_range(p.values[0]) / self.rangeX[donor.first_terminal]

                        if p.arg_range > donor.arg_range or tries < 10:

                            self.bigger_function = (p)
                        if p.arg_range < donor.arg_range or tries < 10: 
                            self.smaller_function = (p)
                        tries += 1

                  
                    
                    
                    
                    if coin > 0.5 : #success for bigger
                     
                      
                        idx_ = self.idx_to_pos[donor.first_terminal][donor_index] 
                        candidate = [ id_ for (id_ , rag_range) in self.terminal_cluster[donor.first_terminal][idx_+1:] ]
                        
                       
                        if self.bigger_function != None:
                            arg_receiver_func = self.bigger_function   
                        elif candidate != []:
                            arg_receiver_idx = random.choice(candidate)
                            receiver = self.programs[arg_receiver_idx]

                        
                   
                           
                        
                    

                    elif coin < 0.5: #success for smaller
                        
                        
                        
                      
                        idx_ = self.idx_to_pos[donor.first_terminal][donor_index] 
                        
                        candidate = [ id_ for (id_ , rag_range) in self.terminal_cluster[donor.first_terminal][:idx_] ]
                        
                   
                        if self.smaller_function != None:
                            arg_receiver_func = self.smaller_function
                          
                        elif candidate != []:
                            arg_receiver_idx = random.choice(candidate)
                            receiver = self.programs[arg_receiver_idx]
                               
                            
                            
                        
                       
                        
                    

                    
                    c = 0
                    

                    if arg_receiver_idx != None:
                        receiver = self.programs[arg_receiver_idx]
                        r_start = receiver.cut_point
                        r_end = receiver.get_subtree(receiver.cut_point)

                        new_program =  donor.program[:start] + receiver.program[r_start:r_end+1] + donor.program[end+1:]

                        c = 1
                        
                    elif arg_receiver_func != None: 
                       
                        new_program =  donor.program[:start] + arg_receiver_func.program + donor.program[end+1:]

                       
                  
                    else:
                        random_func = random.randint(0, len(self.function_set)-1)
                        if self.function_set[random_func].arity == 2:
                            new_program = donor.program[:start] + [ random_func, choice_1, choice_2 ] + donor.program[end+1:]
                        else:
                            new_program = donor.program[:start] + [ random_func, choice_1 ] + donor.program[end+1:]

                    
                    self.next_programs[i] = Program(self.terminal_set , self.function_set , program=new_program)
                    #hahapt
                    self.next_programs[i].fitness = get_fitness_MAD(self.next_programs[i] , self.X , self.y)
                    self.binding_adaptation[ index[i] % 3 ] += self.next_programs[i].fitness
                    # new_diff_ratio = -1
                    # if cal_range(self.next_programs[i].values[0])!=0:
                    #     new_diff_ratio = self.target_range / cal_range(self.next_programs[i].values[0])
                   

                    
                   
                    self.next_programs[i].from_method = 1
               
            elif p < 1: #reproduced
                    donor = self.tournament_selection()
                    self.programs[i] = copy.deepcopy(donor)

 
        self.programs = sorted(self.programs, key=lambda p: p.fitness)
        self.next_programs = sorted(self.next_programs, key=lambda p: p.fitness)
        
        
        
        
        self.bindingFitnessRecord+=[(self.programs[i].rejection, self.programs[i].fitness )
             for i in range(len(self.programs))]
        
        self.programs = self.next_programs
        # mu lambda selection !!!
        candidate = self.programs + self.next_programs
        candidate = sorted(candidate, key=lambda p: p.fitness)
        for i in range(self.num_population):
            #if self.programs[i].fitness > self.next_programs[i].fitness:
                #self.programs[i]._copy(self.next_programs[i])
                self.programs[i]._copy(candidate[i])
        self.programs = sorted(self.programs, key=lambda p: p.fitness)

        # test for ranging 


        # for i in range(1,self.num_population//4):
        #     self.programs[-i] = Program(self.terminal_set , self.function_set) 
        #     self.programs[-i].fitness = get_fitness_MAD(self.programs[-i] , self.X , self.y)

    def run(self):
        self.binding_gen = 10
        self.two_layer_counter = {}
        self.most_two_layer_range_avg = []
        self.least_two_layer_range_avg = []
        self.rejection = [0 for i in range(self.generation)]
        for i in range(len(self.function_set)):
            for j in range(len(self.function_set)):
                for n in range(2):
                    self.two_layer_counter[100*i + 10*j + n] = 0
                
        for g in range(self.generation):
            #self.programs = sorted(self.programs, key=lambda p: p.fitness)
            if True:
            
                self.two_layer_counter_temp = {}
                for k in self.two_layer_counter:
                    self.two_layer_counter_temp[k] = 0
                # learn the binding model  
                
                for p in self.programs[:]:
                    p.intron_removal()
                    p.get_parent_forall()
                    program = p.program
                    
                    for i in range(1,len(program)):
                        if program[i] >= 0: #non-terminal

                            two_layer = get_hash(p,i)
                            
                            self.two_layer_counter_temp[two_layer] += 1                     
                
                for k in self.two_layer_counter:
                    self.two_layer_counter[k] = self.two_layer_counter_temp[k]
            
           
            #s = sum(list(self.two_layer_counter.values()))
            #print("=======")
            #for k in self.two_layer_counter:
            #    print( k,self.two_layer_counter[k] / s)
            # choose the cut point and decide the "funcs" and "args"
            _id = 0
            self.cluster_bigger = [[] , []] 
            self.cluster_smaller = [[], []] 

            big_id = 0
            small_id = 0
            self.terminal_cluster = {}
            self.idx_to_pos = {}
            
            self.rejection[g] = self.chosen_k_largest

            m1 = nlargest( self.chosen_k_largest ,self.two_layer_counter , key=self.two_layer_counter.get )
            self.two_layer_counters[g] = copy.deepcopy(self.two_layer_counter)
            #m_all = nlargest( len(self.two_layer_counter ) ,self.two_layer_counter , key=self.two_layer_counter.get )
            #self.two_layer_counter_for_exp = copy.deepcopy(self.two_layer_counter)
            #print(m_all, [self.two_layer_counter[m] for m in m_all], sum([self.two_layer_counter[m] for m in m_all]))
           
            self.two_layer_counter_binding_prob = [{k:0 for k in self.two_layer_counter},
                                                   {k:0 for k in self.two_layer_counter},
                                                   {k:0 for k in self.two_layer_counter}
                                                   ]
            
            m = [[],[],[]]
            temp = [ max(self.chosen_k_largest-1,0) , self.chosen_k_largest , min(self.chosen_k_largest+1,self.num_two_layer) ]
            #print(m)
            #for k in m :
            #    print(k ,self.two_layer_counter[k] )
   


            
            for i in range(3):
                m[i] = nlargest( temp[i] ,self.two_layer_counter , key=self.two_layer_counter.get )
                m[i] = set(m[i])
                #print(self.two_layer_counter)
                s = 0
                for k in m[i] :
                    s += self.two_layer_counter[k]
                    #print(self.two_layer_counter[k],s,k)
                if s != 0: 
                    for k  in m[i]:
                        self.two_layer_counter_binding_prob[i][k] = copy.deepcopy(self.two_layer_counter[k]/s)
                      
                        # if k in m and self.two_layer_counter[k]/s >  1/len(m[i]):
                        #     self.two_layer_counter_binding_prob[i][k] = 1
                        # else:
                        #     self.two_layer_counter_binding_prob[i][k] = 0

            
            
            index = [i for i in range(len(self.programs))]
            random.shuffle(index)
            _id = 0
            for i in index:
                #print(i, "index")

                p = self.programs[_id]
                
                
                # using binding information

                p.cut_point = p.get_cut_point( self.two_layer_counter_binding_prob[i%3] , 
                                              g , self.binding_gen , self.rejection)
                
                # calculate diff range
                if  abs( max(p.values[0]) - min(p.values[0])) != 0:
                    p.diff_range = self.target_range / abs( max(p.values[0]) - min(p.values[0]))
                else:
                    p.diff_range = 0
                
                # calculate arg range
                p.arg_range = ( abs( max(p.values[p.cut_point]) - min(p.values[p.cut_point]))
                                / abs( 1 ) )  #  output_range  /  1
                
                
                
                first_terminal = None
                for t in p.program[ p.cut_point :  p.get_subtree(p.cut_point)+1  ]:
                    if t < 0:
                        first_terminal = -(t+1)



              
                # take first terminal as input 
                p.first_terminal = first_terminal
                
            
                if first_terminal not in self.terminal_cluster:
                    self.terminal_cluster[first_terminal] = []
                self.terminal_cluster[first_terminal].append( (_id , p.arg_range))

         

                p.get_parent_forall()

              


                #range
                if p.diff_range > 1:

                    self.cluster_bigger[0].append(_id)
                    
                    p.cluster_id = big_id
                    big_id +=1
                else:

                    self.cluster_smaller[0].append(_id)
                    
                    p.cluster_id = small_id
                    small_id+=1

            
                _id += 1

            # sorting for argument range
            for terminal in self.terminal_cluster:
                self.terminal_cluster[terminal] =  sorted(self.terminal_cluster[terminal] , key= lambda p:p[1])
                self.idx_to_pos[terminal] = {}
                for i, (id_ ,_) in enumerate(self.terminal_cluster[terminal]):
                    self.idx_to_pos[terminal][ id_ ] = i
           
     
            self.run_once(index)
            #print(self.binding_adaptation)

            if np.argmax(self.binding_adaptation) == 0:
                self.chosen_k_largest += 1
                self.chosen_k_largest = min( self.chosen_k_largest, self.num_two_layer-1)
            elif np.argmax(self.binding_adaptation) == 1:
                pass
            else:
                self.chosen_k_largest -= 1
                self.chosen_k_largest = max( self.chosen_k_largest, 1 )
            #print(g, self.chosen_k_largest)
            #for i in range(len(self.programs)):
            remove_list = []
            
            # for k in self.two_layer_counter_for_exp:
            #     if self.two_layer_counter_for_exp[k] == 0:
            #         remove_list.append(k)
            # for k in remove_list:
            #      self.two_layer_counter_for_exp.pop(k, None)
 
            # l = nlargest( 1 ,self.two_layer_counter_for_exp , key=self.two_layer_counter_for_exp.get )
            # s = nsmallest( 1 ,self.two_layer_counter_for_exp , key=self.two_layer_counter_for_exp.get )
          
            # most_two_layer = l[0]
            # least_two_layer = s[0]
            # print(most_two_layer , least_two_layer)
            # most_two_layer_range = []
            # least_two_layer_range = []
            # for p in self.programs:
            #     for pt in range(len(p.program)):
            #         if get_hash(p, pt) == most_two_layer:

            #             o = cal_range(p.values[ p.parentIdx[pt] ]) 
                        
            #             i1 = cal_range(p.values[ p.childrenIdx[pt][0]]) 
            #             #if len(p.childrenIdx[pt]) > 0:
            #             #    i2 =  cal_range(p.values[ p.childrenIdx[pt][1]])
            #             r1 = 0
            #             r2 = 0
            #             if i1 != 0:
            #                 r1 = o/i1
                     
            #             #if i2 != 0:
            #             most_two_layer_range.append( r1 )
            #             #print(most_two_layer_range)
            #         if get_hash(p, pt) == least_two_layer:  
            #             o = cal_range(p.values[ p.parentIdx[pt] ]) 
            #             i1 = cal_range(p.values[ p.childrenIdx[pt][0]]) 
            #             #if len(p.childrenIdx[pt]) > 0:
            #             #    i2 =  cal_range(p.values[ p.childrenIdx[pt][1]])
            #             r1 = 0
            #             r2 = 0
            #             if i1 != 0:
            #                 r1 = o/i1
            #             #if i2 != 0:
                   
            #             least_two_layer_range.append( r1 )
           
            # most_two_layer_range.sort()
            # least_two_layer_range.sort()
            # #print(most_two_layer_range)
            # from statistics import geometric_mean

            # ml = len(most_two_layer_range)
            # ll = len(least_two_layer_range)
            # if len(most_two_layer_range[: ml//4 *3]) != 0:
            #     tmp = [ m if m!=0 else 1 for m in most_two_layer_range[: ml//4 *3] ]
            #     #print("most tmp:",tmp , geometric_mean(tmp))
            #     self.most_two_layer_range_avg.append(geometric_mean(tmp) )
            # else:
            #     print( most_two_layer_range )
            #     self.most_two_layer_range_avg.append(0)
            # if len(least_two_layer_range[: ll//4 *3]) != 0:
            #     tmp = [ l if l!=0 else 1 for l in least_two_layer_range[: ll//4 *3] ]
            #     #print("least tmp:",tmp , geometric_mean(tmp))
            #     self.least_two_layer_range_avg.append(geometric_mean(tmp))
            # else:
            #     print( least_two_layer_range )
            #     self.least_two_layer_range_avg.append(0)
            #self._print_programs()


        self.programs = sorted(self.programs, key=lambda p: p.fitness)
    
    
    


    def _print_programs(self):
 
        i = 0
        for p in self.programs:
            print(i, p.fitness) 
            p.printX()
            print()
            i+=1    
    



class Program : #binary version
    def __init__(self , terminal_set  , function_set , depth = 5 ,program = None):
        self.from_method = -1 #init 
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.depth = depth
        self.fitness = None
        self.values = []
        self.cluster_id = -1
        self.rejection = 0
        if program is None :
            self.random_init(self.depth)
        else:
            self.program = program
        # should evaluate when new a program !!!

    def valid(self, program):
        if len(program) == 1 and program[0] < 0 :
            return True
        Counter = [self.function_set[program[0]].arity]

        for p in program[1:]:
  
            
            if p < 0:
                Counter[-1] -= 1
            else:
                Counter[-1] -= 1
                Counter.append( self.function_set[p].arity )
            while Counter !=[] and Counter[-1] == 0:
                Counter.pop()
        return (Counter == [])

    def execute(self,X):
        """
        value  :  tree length list , each element is each input result at i 
        x = [0 , 1]
        [+ x x]
        [[0,2] ,[0,1] ,[0,1]]
        """
        if( not self.valid(self.program)):
            print("WROOOOOONNNNNG")
            return None
        self.terminals = []
        self.range = []
        self.values = []
        output = []

        for p in reversed(self.program):
           
            #print(p , self.program  )
           
                #print("--",i , self.values )
                #print(len(output[i]))
            tmp = None
            if p < 0 : # from terminal set (one dim now!!)
                
                    tmp = X[: , -(p+1)]
                    output.append(tmp)
            else:
                    #print(p,i,output , self.program)
                    if self.function_set[p].arity == 1:
                        tmp = self.function_set[p](self.values[-1])
                        output[-1] = tmp
                    if self.function_set[p].arity == 2:
                        
                        
                        tmp = self.function_set[p](output[-1],output[-2])
                        output[-2] = tmp
                        output.pop()
         
            
            self.values.append(tmp)
          
           

           
        self.values = list(reversed(self.values))
        #print(self.values)
        self.y_hat = self.values[0]

        self.intron_removal()
        self.get_parent_forall()
        
        return self.y_hat
    
    def intron_removal(self , at_i = 0):
    
        # for subtree of subtree
        # ****[---[+++]---]****
        #      ^   ^      ^
        #     at_i new_root     end_i
        if len(self.values) == 1:
            return 
        
        root_value = self.values[at_i]
        #print("==", root_value)
        new_root = at_i
        end_i = self.get_subtree(at_i)
        for i in range(len(self.values[at_i : self.get_subtree(at_i) ])):
            if np.array_equal(self.values[at_i + i],root_value):
                #print(at_i + i)
                new_root = at_i + i
        # update termianls
        
        #print(new_root , len(self.program))

        end = self.get_subtree(new_root)+1
        

        if end_i+1 < len(self.values):
            self.values = self.values[:at_i] + self.values[new_root: end] + self.values[end_i+1:]
            self.program = self.program[:at_i] + self.program[new_root: end] + self.program[end_i+1:]
        else:
            self.values = self.values[:at_i] + self.values[new_root: end] 
            self.program = self.program[:at_i] + self.program[new_root: end] 
        
    def get_parent_forall(self):
        self.parentIdx = [ -1 for i in range(len(self.program))]
        self.nthChild = [-1 for i in range(len(self.program))]
        self.childrenIdx = [[] for i in range(len(self.program))]
        if len(self.program) == 1:
            return 
        
        IdxStack = []
        remain = []
        for i in range(len(self.program)):
            if self.program[i] < 0:
                self.parentIdx[i] = IdxStack[-1] 
                remain[-1] -= 1
                while remain[-1] == 0:
                    remain.pop()
                    IdxStack.pop()
                    if len(remain) == 0:
                        break
                    remain[-1] -= 1
            else:
                if IdxStack != []:
                    self.parentIdx[i] = IdxStack[-1] 
                IdxStack.append(i)
                remain.append( self.function_set[ self.program[i] ].arity )
        table = {}
        for i in range(len(self.program)):
            if self.parentIdx[i] < 0: # for root
                continue
            if self.parentIdx[i] not in table:
                table[self.parentIdx[i]] = 0
            else:
                table[self.parentIdx[i]] += 1
            self.nthChild[i] = table[self.parentIdx[i]]
        
        for i in range(len(self.program)):
            if self.parentIdx[i] != -1:
                self.childrenIdx[ self.parentIdx[i] ].append(i)
        
    """
    def get_path_range(self,i):
        cur = i
        res = 1
        #print(len(self.parentIdx) , cur)
        while self.parentIdx[cur] != -1:
           
            parent = self.parentIdx[cur]

            if abs( max(self.values[cur]) - min(self.values[cur]) )!= 0:
                res*= ( 
                    abs( max(self.values[parent]) - min(self.values[parent]))
                    / abs( max(self.values[cur]) - min(self.values[cur]) ))
            else:
                res *= 0

            cur = parent 
            
        return res
    """
    def get_path_range(self,i):
       
        res = 0
        n = abs( max(self.values[i]) - min(self.values[i]) )
        if n!= 0:
            res = abs( max(self.values[0]) - min(self.values[0]) ) /n
            
        return res

    def random_init(self , depth = 3):
        t = len(self.terminal_set)
        f = len(self.function_set)
        #ramp half and half 
        p = random.uniform(0, 1)
        self.program = []
        if p < 0.5:# full
            d = 0
            Counter = []

            choice = random.randint(0,f-1)
            self.program.append(choice)
            
            Counter.append(self.function_set[choice].arity)
            while (Counter != []):
                if len(Counter) < depth :
                    Counter[-1] -= 1
                    choice = random.randint(0,f-1)
                    self.program.append(choice)
                    Counter.append(self.function_set[choice].arity)
                else:
                    Counter[-1] -= 1
                    choice = -1 * random.randint(1,t)
                    self.program.append(choice)
                    while Counter[-1] == 0:
                        Counter.pop()
                        if len(Counter) == 0:
                            break

        else:
            d = 0
            not_terminal = True
            Counter = []

            choice = random.randint(0,f-1)
            self.program.append(choice)
     
            Counter.append(self.function_set[choice].arity)
            while (Counter != []) :
                
                if len(Counter) < depth:
                    p = random.uniform(0, 1)
                else:
                    p = 1 # must terminate !! 
                if p < 0.9: #function
                    Counter[-1] -= 1
                    choice = random.randint(0,f-1)
                    self.program.append(choice)
                    Counter.append(self.function_set[choice].arity)
                else:
                    Counter[-1] -= 1
                    self.program.append(-1 * random.randint(1,t))
                    while Counter[-1] == 0:
                        Counter.pop()
                        if len(Counter) == 0:
                            break
                

    def printX(self):
        if len(self.program) == 1:
            print("X" + str(-(self.program[0]+1)) , end=" ")
            return 
        Counter = []
        for p in self.program:
            if p < 0:
                Counter[-1] -= 1
                print("X" + str(-(p+1)) , end=" ")
            else:
                if len(Counter) != 0 :
                    Counter[-1] -= 1
                Counter.append(self.function_set[p].arity)
                print(self.function_set[p].name , "(", end=" ")
            while Counter != [] and Counter[-1] == 0:
                Counter.pop()
                print(")",end=" ")


    def get_subtree(self , i):
   
        if self.program[i] < 0:
            return i
        counter = self.function_set[self.program[i]].arity 
        cur = i
       
        while counter != 0:
            #print(counter,self.program[cur])
            cur += 1
            counter -= 1
            if self.program[cur] >= 0:
                counter += self.function_set[self.program[cur]].arity 
        return cur
    def _copy(self,p):
        self.program = copy.deepcopy(p.program)
        self.fitness = p.fitness
        self.values = copy.deepcopy(p.values)
        self.parentIdx = copy.deepcopy(p.parentIdx)
        self.nthChild = copy.deepcopy(p.nthChild)
        self.childrenIdx  = copy.deepcopy(p.childrenIdx)

    def get_cut_point( self , prob , generation , bind_gen, rejection_record ,mode=1 ):
        self.rejection = 0
        for i in range(1,len(self.program)):
            if self.program[i] >= 0: #non-terminal

                two_layer = get_hash(self,i)
                            
                if two_layer in prob and prob[two_layer]>0:
                    self.rejection += 1
        #print(self.rejection)
        
        if mode == 0 :
            pt = random.randint(0,len(self.program)-1)
        else:
           
            
            pt = random.randint(0,len(self.program)-1)
            hash_code = get_hash(self,pt)
            if hash_code in prob:
                
                coin = np.random.choice([1,0], p=[prob[hash_code] , 1-prob[hash_code]])
            else: #terminal
                coin = np.random.choice([1,0], p=[0 , 1])
            #print(coin)
            while (coin == 1 and 
                  pt != 0) :
                
                #print(self.rejection)
                rejection_record[-1] += 1
                pt = random.randint(0,len(self.program)-1)
                hash_code = get_hash(self,pt)
                if hash_code in prob:
                    coin = np.random.choice([1,0], p=[prob[hash_code] , 1-prob[hash_code]])
                else:#terminal
                    coin = np.random.choice([1,0], p=[0, 1])
            
        return pt
            
        
