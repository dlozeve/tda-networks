#!/usr/bin/env python3

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import OneClassSVM

import dill

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (10, 6)


N_CLUSTERS = 10

GENERATIVE = True
SOCIOPATTERNS = True

if __name__=="__main__":
    if GENERATIVE:
        print("==== Generative model ====")
        zz_dgms = dill.load(open("generative/zz_dgms.dill", "rb"))
        wrcf_dgms = dill.load(open("generative/wrcf_dgms.dill", "rb"))
        zz_gram1 = dill.load(open("generative/zz_gram1.dill", "rb"))
        wrcf_gram1 = dill.load(open("generative/wrcf_gram1.dill", "rb"))
        zz_distmat = dill.load(open("generative/zz_distmat.dill", "rb"))
        wrcf_distmat = dill.load(open("generative/wrcf_distmat.dill", "rb"))

        print("Zigzag + kernel")
        clf = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(zz_gram1)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/gen_zz_k.pdf", transparent=True,
                    pad_inches=0.3, bbox_inches="tight")

        print("WRCF + kernel")
        clf = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(wrcf_gram1)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/gen_wrcf_k.pdf", transparent=True,
                    pad_inches=0.3, bbox_inches="tight")
        
        print("Zigzag + bottleneck")
        clf = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(zz_distmat)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/gen_zz_b.pdf", transparent=True,
                    pad_inches=0.3, bbox_inches="tight")

        print("WRCF + bottleneck")
        clf = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(wrcf_distmat)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/gen_wrcf_b.pdf", transparent=True,
                    pad_inches=0.3, bbox_inches="tight")

    if SOCIOPATTERNS:
        print("==== SocioPatterns dataset ====")
        zz_dgms = dill.load(open("sociopatterns/zz_dgms.dill", "rb"))
        wrcf_dgms = dill.load(open("sociopatterns/wrcf_dgms.dill", "rb"))
        zz_gram1 = dill.load(open("sociopatterns/zz_gram1.dill", "rb"))
        wrcf_gram1 = dill.load(open("sociopatterns/wrcf_gram1.dill", "rb"))
        zz_distmat = dill.load(open("sociopatterns/zz_distmat.dill", "rb"))
        wrcf_distmat = dill.load(open("sociopatterns/wrcf_distmat.dill", "rb"))
        
        print("Zigzag + kernel")
        clf = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(zz_gram1)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/sp_zz_k.pdf", transparent=True, pad_inches=0.3, bbox_inches="tight")
        
        print("WRCF + kernel")
        clf = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(wrcf_gram1)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/sp_wrcf_k.pdf", transparent=True, pad_inches=0.3, bbox_inches="tight")
        
        print("Zigzag + bottleneck")
        clf = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(zz_distmat)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/sp_zz_b.pdf", transparent=True, pad_inches=0.3, bbox_inches="tight")
        
        print("WRCF + bottleneck")
        clf = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
        clf.fit(wrcf_distmat)
        fig, ax = plt.subplots()
        ax.step(range(len(clf.labels_)), clf.labels_, where='post')
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Cluster")
        fig.savefig("fig/sp_wrcf_b.pdf", transparent=True, pad_inches=0.3, bbox_inches="tight")
