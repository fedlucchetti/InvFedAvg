import sys,os,json
from fedlearning import bcolors
from tensorflow.math import reduce_sum

# import bcolors
from os.path import join,split
from pathlib import Path
bcolors = bcolors.bcolors()
from fnmatch import fnmatch
import numpy as np
import pyemd



class Utils(object):
    def __init__(self):
        file_path           = os.path.realpath(__file__)
        self.HOMEDIR        = split(split(split(file_path)[0])[0])[0]
        self.DATADIR        = join(self.HOMEDIR ,'data')
        self.DATASETDIR     = join(self.DATADIR ,'training','distributed')

    def create_dir(self,dirpath):
        os.makedirs(dirpath, exist_ok=True)

    def printout(self,message,color=bcolors.HEADER):
        print(color)
        print("#"*160)
        print("#"*int((160-len(message))/2),message,"#"*int((160-len(message))/2))
        print("#"*160)
        print(bcolors.ENDC)


    def list_history_files(self,folder,pattern="*.json"):
        outfiles=[]
        for subpath, subdirs, files in os.walk(folder):
            for filename in files:
                if fnmatch(filename, pattern) and 'history' in filename and "round" not in subpath:
                    outfiles.append(os.path.join(subpath,filename))
        return outfiles

    def list_curve_files(self,folder,pattern="*.json"):
        outfiles=[]
        for subpath, subdirs, files in os.walk(folder):
            for filename in files:
                if fnmatch(filename, pattern) and 'curve' in filename and "round" not in subpath:
                    outfiles.append(os.path.join(subpath,filename))
        return outfiles

    def get_curve_results(self,emd=0,nclients=5,algorithm="fedoutavg",dataset="cifar100",batch_size=8,local_epochs=5,learning_rate=0.001,mu=0.0):
        emd_array,loss_array,metric_array,nclient_array=[],[],[],[]
        history_files = self.list_curve_files(join("results",algorithm,dataset),"*.json")
        print("Found", len(history_files),"curve files")
        for file in history_files:
            if "C"+str(nclients) in file and "single" not in file and str(emd) in file and \
            str(batch_size) in file and str(local_epochs) in file and \
            str(learning_rate) in file and str(mu) in file:
                print("Found file",file)
                with open(file) as fileid:
                    data=json.load(fileid)
        return data

    def get_results(self,emd=0,nclients=5,algorithm="fedoutavg",dataset="cifar100",batch_size=8,local_epochs=5,learning_rate=0.001,mu=0.0):
        emd_array,loss_array,metric_array,nclient_array=[],[],[],[]
        history_files = self.list_history_files(join("results",algorithm,dataset),"*.json")
        print("Found", len(history_files),"history files")
        for file in history_files:
            if "C"+str(nclients) in file and "single" not in file and str(emd) in file and \
            str(batch_size) in file and str(local_epochs) in file and \
            str(learning_rate) in file and str(mu) in file:
                print("Found file",file)
                with open(file) as fileid:
                    data=json.load(fileid)
                # try:
                if data["metric"][0]> 0.01:
                    print(file,data["metric"][0])
                    emd_array.append(np.mean(data["emd"]))
                    loss_array =data["val_loss"]
                    metric_array.append(data["metric"][0])
                    nclient_array.append(data["options"]["n_clients"])
                # except Exception as e:print("get results Exception",e)
        emd_array = np.array(emd_array)
        emd_array = np.reshape(emd_array,[emd_array.shape[0],1])

        loss_array = np.array(loss_array)
        # loss_array = np.reshape(loss_array,[loss_array.shape[0],1])
        return emd_array, loss_array, metric_array, nclient_array

    def shuffle_set(self,X,Y):
        randid   = np.random.randint(0,X.shape[0],X.shape[0])
        X        = X[randid,:,:]
        Y        = Y[randid]
        return X, Y

    def format_genomic_label(self,labels,sensposition):
        outlabels = np.zeros([labels.size,150])
        for id,label in enumerate(labels):
            if label==1:
                outlabels[id,sensposition] = 1
        return outlabels

    def scale_model_weights(self,weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        for i in range(len(weight)):
            weight_final.append(scalar * weight[i])
        return weight_final

    def sum_scaled_weights(self,scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad

    def add2cluster(self,clusters,id1,id2):
        flag=0
        if len(clusters)==0:
            clusters.append({id1,id2})
            return clusters
        for cluster_id in range(len(clusters)):
            if not clusters[cluster_id].isdisjoint({id1,id2}):
                clusters[cluster_id].update({id1,id2})
            else:
                flag+=1
        if flag==len(clusters):
            clusters.append({id1,id2})
        return clusters

    def add_singlton2cluster(self,clusters,nclients):
        for id in range(nclients):
            flag=0
            for cluster in clusters:
                if id not in cluster:
                    flag+=1
            if flag==len(clusters):
                clusters.append({id})
        return clusters



    def remove_duplicate(self,clusters):
        id2remove,_clusters=[],[]
        for test_id in range(len(clusters)):
            for id in range(test_id+1,len(clusters)):
                if clusters[test_id] == clusters[id]:
                    id2remove.append(id)
        id2remove=np.unique(id2remove)
        for id,cluster in enumerate(clusters):
            if id not in id2remove:
                _clusters.append(cluster)
        return _clusters

    def vectorize(self,read,word_size=5):
        vector_word=""
        for id,cursor in enumerate(range(0,len(read),word_size)):
            vector_word+=read[cursor:cursor+word_size]+" "
        return vector_word

    def split_noniid(self,train_idcs, train_labels, alpha, nclients):
        '''
        Splits a list of data indices with corresponding labels
        into subsets according to a dirichlet distribution with parameter
        alpha
        '''
        n_classes = int(train_labels.max()+1)
        label_distribution = np.random.dirichlet([alpha]*nclients, n_classes)
        class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten()
               for y in range(n_classes)]
        client_idcs = [[] for _ in range(nclients)]
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]
        client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
        return client_idcs

    #
    def split_noniid_multiclass(self,train_idcs,feature_labels, train_labels, alpha, nclients):
        '''
        Distributes features and labels among nclients following a Dirichlet
        distribution with parameter alpha
        '''
        labels = np.array([feature_labels,train_labels.flatten()]).T
        classes = []
        for i in range(int(max(train_labels.flatten()))):
            for j in range(int((max(feature_labels))+1)):
                classes.append([i,j])
        classes = np.array(classes)
        n_classes = classes.shape[0]
        label_distribution = np.random.dirichlet([alpha]*nclients, n_classes)
        # class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() for y in classes]
        class_idcs      =[]
        idc_match_pos0 = [np.argwhere(labels[train_idcs][:,0]==y[0]).flatten() for y in classes]
        idc_match_pos1 = [np.argwhere(labels[train_idcs][:,1]==y[1]).flatten() for y in classes]
        ensemble0      = [np.zeros(labels[train_idcs].shape[0]) for y in classes]
        ensemble1      = [np.zeros(labels[train_idcs].shape[0]) for y in classes]
        for id,y in enumerate(classes):
            ensemble0[id][idc_match_pos0[id]]=1
            ensemble1[id][idc_match_pos1[id]]=1
            class_idcs.append(np.where(ensemble0[id]+ensemble1[id]==2)[0])

        client_idcs = [[] for _ in range(nclients)]
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]
        client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
        return client_idcs

    def create_run_id(self,dir):
        ids=[]
        for _subdir in os.listdir(dir):
            print("utils.create_run_id",_subdir[4::])
            ids.append(int(_subdir[4::]))
        ids=np.array(ids)
        while True:
            id_new = np.random.randint(9999)
            if id_new not in ids:
                break
        return id_new

    def gamma_curve(self,u,theta, w1, w2,mode="bezier2"):
        gamma=[]
        if mode=="chain":
            for _theta, _w1, _w2 in zip(theta,w1, w2):
                if u>=0 and u <=0.5: gamma.append(2*(u*_theta+(0.5-u)*_w1 ))
                else:                gamma.append(2*((u-0.5)*_w2 + (1-u)*_theta))
        elif mode=="bezier2":
            for _theta, _w1, _w2 in zip(theta, w1,w2):
                gamma.append( (1-u)**2 * _w1 + 2*u*(1-u)*_theta + u**2*_w2 )
        elif mode=="bezier3":
            theta1,theta2 = theta
            for _theta1,_theta2, _w1, _w2 in zip(theta1,theta2, w1,w2):
                gamma.append( (1-u)**3 * _w1 + 3*u*(1-u)**2*_theta1 + 3*(1-u)*u**2 * _theta2 + u**3*_w2 )
        return gamma

    def jacobian_dgamma(self,u,mode,var=1):
        if mode=="chain":
            if u>=0 and u <=0.5: return u
            else:                return 1-u
        elif mode=="bezier2":
            return u*(1-u)
        elif mode=="bezier3":
            if var==1:
                return  3*u*(1-u)**2
            elif var==2:
                return  3*(1-u)*u**2




if __name__ == '__main__':
    utils=Utils()
    print(utils.list_history_files("results","*.json"))
