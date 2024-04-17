# IMPORTS
#%pip install faiss
import numpy as np
import h5py
import PartitionTreesOriginal as PT

#import faiss
import random
import time
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

text='output1000queries.txt'




def save_results(results, file):
    text=""
    with open(file, 'a') as file:
        for result in results:
            text += f"{result}"
            text += ";"
        text +="\n"
        file.write(text)

#import matplotlib.pyplot as plt
# ---------- Data Variables ----------
for sampleMethod in ["random", "fullrptree"]:
    for DATA in  ["MNIST", "GloVe"]:
        for APPROXIMATETRUTH in [True,False]:
            LEAFSIZE = 50
            K = 10
            FORESTSIZE = 0
            QUERYINDEX = 1
            VOTETHRESHOLD = 1

            # ---------- dependent variables ----------
            if DATA =="MNIST":
                ANGULARDISTANCE = False
                NORMALIZEDDATA = False
                METRIC='euclidean'
                NCLUSTERS = [1,5,10,15]
            if DATA == "GloVe":
                ANGULARDISTANCE = True
                NORMALIZEDDATA = True
                METRIC='cosine'
                NCLUSTERS = [1,5,20,50]
            # #STANDARD DATASET
            # IMPORT DATASET
            if DATA == "GloVe":
                glove_h5py = h5py.File('glove-100-angular.hdf5', "r")

                if APPROXIMATETRUTH:
                    resultFile = h5py.File('hnswef5M5_glove.h5', "r")
                else:
                    resultFile = h5py.File('glove-100-angular-groundtruth.h5', "r")
            if DATA == "MNIST":
                glove_h5py = h5py.File('fashion-mnist-784-euclidean.hdf5', "r")

                if APPROXIMATETRUTH:
                    resultFile = h5py.File('hnswef5M5_mnist.h5', "r")
                else:
                    resultFile = h5py.File('fashion-mnist-784-euclidean-groundtruth.h5', "r")




            groundTruth = resultFile['neighbors'][:]
            dataset = glove_h5py['train'][:] ## the final [:]-slice is necessary to turn the data into a numpy-array, from the .h5py format
            queries = glove_h5py['test'][:]
            neighborsFromQuery = glove_h5py['neighbors'][:]  
            distancesFromQuery = glove_h5py['distances'][:]


            if NORMALIZEDDATA:
                #Normalization
                dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
                queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]

            partitionsize = LEAFSIZE
            forest=[]
            if FORESTSIZE>0:
                for i in range(int(np.ceil(FORESTSIZE/10))):
                    filename = 'partitionsize_'+str(partitionsize) +'_forestno_' + str(i) + '.pkl'
                    with open(filename, 'rb') as file:
                        loaded_data = pickle.load(file)
                    forest = forest + loaded_data
                forest = forest[:FORESTSIZE]



            def distance(a, b):
                    if ANGULARDISTANCE and NORMALIZEDDATA:
                        #Angular Distance - Normalized
                        return np.arccos(np.dot(a, b))/np.pi
                    elif ANGULARDISTANCE:
                        #Angular Distance - Non-normalized
                        return np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))/np.pi
                    else:
                        #Euclidean Distance 
                        return np.linalg.norm(a-b)

            def gmm_and_cluster_sampling(dataset,components,samples,sampleMethod):
            ##################################################################################################################################################
                def gaussian_mixture_model_and_sampling(dataset,components,samples):
                    gmm = GaussianMixture(n_components=components)  
                    gmm.fit(dataset)
                    clusters={}
                    cnt=0
                    indices = [i for i in range(len(dataset))] #list(range(len(my_list)))
                    random.shuffle(indices)
                    for point in indices:
                        probabilities = gmm.predict_proba(np.array([dataset[point]]))
                        most_likely_cluster_index = np.argmax(probabilities)
                        if most_likely_cluster_index not in clusters:
                            clusters[most_likely_cluster_index] = []
                        if len(clusters[most_likely_cluster_index])==samples:
                            cnt+=1
                        if cnt==components:
                            break
                        clusters[most_likely_cluster_index].append(point)
                    for i in range(components):
                        clusters[i]=clusters[i][:samples]
                    return gmm, clusters
            ##################################################################################################################################################
                def gaussian_samples(dataset,components,samples):
                    gmm = GaussianMixture(n_components=components)  
                    gmm.fit(dataset)
                    clusters={}
                    cnt=0
                    indices = [i for i in range(len(dataset))] #list(range(len(my_list)))
                    random.shuffle(indices)
                    for point in indices:
                        probabilities = gmm.predict_proba(np.array([dataset[point]]))
                        most_likely_cluster_index = np.argmax(probabilities)
                        if most_likely_cluster_index not in clusters:
                            clusters[most_likely_cluster_index] = []
                        if len(clusters[most_likely_cluster_index])==samples:
                            cnt+=1
                        if cnt==components:
                            break
                        clusters[most_likely_cluster_index].append(point)
                    samples = set()
                    for i in range(components):
                        for sample in clusters[i][:samples]:
                            samples.add(sample)
                    return False, list(samples)
            ##################################################################################################################################################
                def gaussian_mixture_model_and_clustertrees(dataset,components,samples):
                    gmm = GaussianMixture(n_components=components)  
                    gmm.fit(dataset)
                    clusters={}
                    cnt=0
                    indices = [i for i in range(len(dataset))] #list(range(len(my_list)))
                    random.shuffle(indices)
                    for point in indices:
                        probabilities = gmm.predict_proba(np.array([dataset[point]]))
                        most_likely_cluster_index = np.argmax(probabilities)
                        if most_likely_cluster_index not in clusters:
                            clusters[most_likely_cluster_index] = []
                        clusters[most_likely_cluster_index].append(point)
                    for i in range(components):
                        clusters[i]=PT.rp_tree(dataset=dataset[clusters[i]], maxSize=samples)
                    return gmm, clusters
            ##################################################################################################################################################
                def gaussian_mixture_model_and_fullrpclustertrees(dataset,components,samples):
                    gmm = GaussianMixture(n_components=components)  
                    gmm.fit(dataset)
                    clusters={}
                    cnt=0
                    indices = [i for i in range(len(dataset))] #list(range(len(my_list)))
                    random.shuffle(indices)
                    for point in indices:
                        probabilities = gmm.predict_proba(np.array([dataset[point]]))
                        most_likely_cluster_index = np.argmax(probabilities)
                        if most_likely_cluster_index not in clusters:
                            clusters[most_likely_cluster_index] = []
                        clusters[most_likely_cluster_index].append(point)
                    for i in range(components):
                        clusters[i]=PT.rp_tree_full(dataset=dataset[clusters[i]], maxSize=samples, randomized=True)
                    return gmm, clusters
            ##################################################################################################################################################
                if sampleMethod=="random":
                    return gaussian_mixture_model_and_sampling(dataset,components,samples)
                if sampleMethod=="rptree":
                    return gaussian_mixture_model_and_clustertrees(dataset,components,samples)
                if sampleMethod=="fullrptree":
                    return gaussian_mixture_model_and_fullrpclustertrees(dataset,components,samples)
                if sampleMethod=="clusterfreesamples":
                    return gaussian_samples(dataset,components,samples)
                else: 
                    return False
                


            ##################################################################################################################################################
            ##################################################################################################################################################
            ##################################################################################################################################################
                


            def annuli_trajecting(gmm,clusters,queryPoint,kå, annulusSize,annulusInnerCircleSize, annulusCores, sampleMethod):
            ##################################################################################################################################################
            #                                       Define the searching methods
            ##################################################################################################################################################
                def rp_tree_naive_traversal(tree, queryPoint):
                    node = tree
                    while not node.isLeaf:
                        vectorDimensions = node.vectorDimensions
                        randomVector = node.randomVector
                        medianValue = node.splitValue
                        if PT.projection_factor(pointVector=queryPoint, randomVectorDimensions=vectorDimensions, randomVector=randomVector) < medianValue:
                            node = node.leftChild
                        else:
                            node = node.rightChild
                    return node.points
            ##################################################################################################################################################
                def rp_tree_full_traversal(tree, queryPoint):
                    node=tree

                    while not node.isLeaf:
                        randomVector = node.randomVector
                        medianValue = node.splitValue
                        distanceToPlane = np.dot(queryPoint, randomVector)
                    
                        if distanceToPlane < medianValue:
                            node = node.leftChild
                        else:
                            node = node.rightChild
                    return node.points
            ##################################################################################################################################################
                def ENN_search(points, querypoint, k):
                    pointsSearched.update(points)
                    neighbors = []
                    for point in points:
                        neighbors.append((point, distance(dataset[point], querypoint)))
                    neighbors.sort(key= lambda a : a[1])
                    neighbors = [neighbor[0] for neighbor in neighbors]
                    return neighbors[:k]
                
            ##################################################################################################################################################
                def cluster_containing(queryPoint,gmm,clusters):
                    if sampleMethod=="clusterfreesamples":
                        return clusters, 0
                    probabilities = gmm.predict_proba(np.array([queryPoint]))
                    most_likely_cluster_index = np.argmax(probabilities)
                    clusterProbability = probabilities[0][most_likely_cluster_index]
                    if sampleMethod=="random":
                        cluster = clusters[most_likely_cluster_index]
                    if sampleMethod=="rptree":
                        cluster = rp_tree_naive_traversal(tree=clusters[most_likely_cluster_index], queryPoint=queryPoint)
                    if sampleMethod=="fullrptree":
                        cluster = rp_tree_full_traversal(tree=clusters[most_likely_cluster_index], queryPoint=queryPoint)
                            
                    return cluster, clusterProbability
                
            ##################################################################################################################################################
                def moldNeighborhood(oldCand, newCand,cores,size,k):
                    neighbors = set()
                    checkedCandidates = set(oldCand) #| set(newCand)#remove newcand if add in loop TEST MORE HERE PLS
                    for candidate in newCand[:cores]:#round(len(newCand)*focus)]:#newCand[:min(len(newCand),focus)] can also be used TEST MORE HERE PLS
                        candidateNeighbors = groundTruth[candidate,annulusInnerCircleSize:annulusInnerCircleSize+size]#kan ændre k
                        neighbors.update(candidateNeighbors)
                        checkedCandidates.add(candidate)#if remove add to checked before Loop
                    neighbors = checkedCandidates|neighbors|set(newCand)
                    neighbors = ENN_search(list(neighbors), queryPoint, k)
                    oldC = []
                    newC = []
                    for neighbor in neighbors:
                        if neighbor in checkedCandidates:
                            oldC = oldC + [neighbor]
                        else:
                            newC = newC + [neighbor]
                    return oldC, newC

            ##################################################################################################################################################
            #                                       Traverse the partition tree and perform a neighbor search in the leafpoints
            ##################################################################################################################################################
                k=kå
                pointsSearched = set()
                coreNeighbors = max(kå,annulusCores)
                clusterSample, clusterProbability = cluster_containing(queryPoint,gmm,clusters)
                initialBall = ENN_search(clusterSample, queryPoint, coreNeighbors)

            ##################################################################################################################################################
            #                                       Search for neighborhood around q
            ##################################################################################################################################################
                checkedNeighborhood = []
                uncheckedNeighborhood = np.copy(initialBall)
                cnt=0  
                while len(uncheckedNeighborhood) > 0:
                    checkedNeighborhood, uncheckedNeighborhood = moldNeighborhood(checkedNeighborhood,uncheckedNeighborhood,annulusCores,annulusSize,coreNeighbors)
                    cnt+=1

            ##################################################################################################################################################
            ##################################################################################################################################################
                return checkedNeighborhood[:kå], len(pointsSearched), cnt, clusterProbability




            
            def querie_samples(queries, numberOfQueries):
                qIndices = [i for i in range(len(queries))] #list(range(len(my_list)))
                random.shuffle(qIndices)
                querySamples = np.array([queries[i] for i in qIndices[:numberOfQueries]])
                return querySamples



            def get_clusterspan(queryPoint,neighbors,gmm):
                clusterSpan = set()
                probabilities = gmm.predict_proba(np.array([queryPoint]))
                most_likely_query_cluster_index = np.argmax(probabilities)
                clusterSpan.add(most_likely_query_cluster_index)
                neighbors_in_query_cluster=0
                for neighbor in neighbors:
                    probabilities = gmm.predict_proba(np.array([dataset[neighbor]]))
                    most_likely_cluster_index = np.argmax(probabilities)
                    if most_likely_cluster_index==most_likely_query_cluster_index:
                        neighbors_in_query_cluster+=1
                    clusterSpan.add(most_likely_cluster_index)
                return len(clusterSpan), neighbors_in_query_cluster


            save_results(["recall","runtime","algorithm","dataset","approximateTraining","neighborhoodSize","numberOfNeighborhoods","pointsSearched","iterations","clusterProbability","numberOfClusters","sampleMethod", "nSamples","kå","clusterSpan","queryClusterNeighbors","successfulQueries"],text)
            random.seed(95)
            testQueryVectors = querie_samples(queries,1000)
            for samples in [100,1000]:
                #for sampleMethod in ["random", "rptree", "fullrptree"]:
                    for nClusters in NCLUSTERS:
                        np.random.seed(42)
                        random.seed(42)
                        gmm, clusters = gmm_and_cluster_sampling(dataset,nClusters,samples,sampleMethod)
                        for kå in [10,50]:
                            nbrs = NearestNeighbors(n_neighbors=kå, algorithm='auto', metric=METRIC).fit(dataset)
                            for annSize in [10,50,100]:
                                for annCores in [1,2]:
                                    for annulusInnerCircleSize in [0]:  
                                        results1 = []
                                        times = []
                                        npoints = 0
                                        niterations = 0
                                        clusterProbability = 0
                                        cspan=0
                                        qclusterneighbors=0
                                        totalRecall = 0
                                        totalTime = 0
                                        successfulQueries=0
                                        def timer(queryVector,annSize,annCores):
                                            start = time.time()
                                            result, npo, nit, cprob = annuli_trajecting(gmm=gmm,clusters=clusters,queryPoint=queryVector,kå=kå, annulusSize=annSize,annulusInnerCircleSize=annulusInnerCircleSize, annulusCores=annCores,sampleMethod=sampleMethod)
                                            end = time.time()
                                            cs,qcn=get_clusterspan(queryVector,result,gmm)
                                            timepassed = (end - start)
                                            return result, timepassed, npo, nit, cprob, cs, qcn
                                        for queryVector in testQueryVectors:
                                            result, timepassed, npo, nit, cprob, cs, qcn= timer(queryVector=queryVector,annSize=annSize,annCores=annCores)
                                            results1.append([j for j in result])
                                            times.append(timepassed)
                                            if len(result)==kå:
                                                _, indices = nbrs.kneighbors(queryVector)
                                                recall=0
                                                successfulQueries+=1
                                                for neib in result:
                                                    if neib in indices[0]:
                                                        recall+=1
                                                recall=recall/kå
                                                totalRecall+=recall
                                                totalTime = totalTime + float(timepassed)
                                                npoints+=npo
                                                niterations+=nit
                                                cspan+=cs
                                                qclusterneighbors+=qcn
                                                clusterProbability+=cprob
                                        save_results([totalRecall/successfulQueries,totalTime/successfulQueries,"trajecting_neighborhoods_gmm_clustering",DATA,APPROXIMATETRUTH,annSize,annCores,npoints/successfulQueries,niterations/successfulQueries,clusterProbability/successfulQueries,nClusters,sampleMethod,samples,kå,cspan/successfulQueries,qclusterneighbors/successfulQueries,successfulQueries],text)