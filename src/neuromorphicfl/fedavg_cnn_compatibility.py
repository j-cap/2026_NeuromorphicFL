from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .fmnist_multiclass_benchmark import MulticlassFederation
from .fmnist_cnn_benchmark import LAYOUT, CompactCNNLayout, initialize_cnn, loss_and_gradient, predictive_metrics
from .fedavg_event_compatibility import FedAvgConfig


def _local_delta(w_server, X, y, *, config: FedAvgConfig, rng, layout: CompactCNNLayout):
    w_local=w_server.copy()
    n=len(y)
    for _ in range(config.local_steps):
        ids=rng.integers(0,n,size=config.batch_size)
        _,_,g=loss_and_gradient(w_local,X[ids],y[ids],layout=layout,regularization=config.regularization,need_gradient=True)
        w_local -= config.local_lr*g
    return (w_local-w_server).astype(np.float32,copy=False)


def run_fedavg_cnn(*,federation: MulticlassFederation,method: str,config: FedAvgConfig,seed: int=70707,layout: CompactCNNLayout=LAYOUT,init_scale: float=0.5):
    rng=np.random.default_rng(seed)
    n_clients=federation.n_clients
    d=layout.dimension
    address_bits=int(math.ceil(math.log2(d)))
    event_bits=address_bits+1
    topk=max(1,int(round(config.topk_fraction*d)))
    w=initialize_cnn(layout=layout,seed=7777,scale=init_scale)
    membrane=np.zeros((n_clients,d),dtype=np.float32)
    residual=np.zeros((n_clients,d),dtype=np.float32)
    last_sign=np.zeros((n_clients,d),dtype=np.int8)
    payload=0; packetized=0; messages=0; events=0; reversals=0; repeated=0
    delta_norms=[]; membrane_norms=[]; history=[]
    event_gain=config.encoder_gain_multiplier/config.local_lr
    weights=federation.weights.astype(float); weights/=weights.sum()

    for rnd in range(1,config.rounds+1):
        deltas=[]
        for i in range(n_clients):
            dlt=_local_delta(w,federation.client_X[i],federation.client_y[i],config=config,rng=rng,layout=layout)
            deltas.append(dlt); delta_norms.append(float(np.linalg.norm(dlt)))
        if method=="dense_fedavg":
            agg=np.zeros(d,dtype=np.float32)
            for i in range(n_clients):
                agg += weights[i]*deltas[i]
                bits=32*d; payload+=bits; packetized+=bits+64; messages+=1
            w += agg
        elif method=="ef_topk_fedavg":
            agg=np.zeros(d,dtype=np.float32)
            for i in range(n_clients):
                residual[i] += weights[i]*deltas[i]
                ids=np.argpartition(np.abs(residual[i]),-topk)[-topk:]
                vals=residual[i,ids].copy(); agg[ids]+=vals; residual[i,ids]=0
                bits=topk*(32+address_bits); payload+=bits; packetized+=bits+64; messages+=1
            w += agg
        elif method=="sign_ef_fedavg":
            agg=np.zeros(d,dtype=np.float32)
            for i in range(n_clients):
                residual[i] += weights[i]*deltas[i]
                scale=float(np.mean(np.abs(residual[i])))
                if scale>0:
                    comp=(scale*np.sign(residual[i])).astype(np.float32)
                    agg += comp; residual[i]-=comp
                bits=d+32; payload+=bits; packetized+=bits+64; messages+=1
            w += agg
        elif method=="event_fedavg":
            membrane *= config.rho
            agg=np.zeros(d,dtype=np.float32)
            jump=config.jump0*(1+rnd/config.jump_scale)**(-config.jump_exponent)
            for i in range(n_clients):
                membrane[i] += event_gain*weights[i]*deltas[i]
                mask=np.abs(membrane[i])>=config.threshold
                count=int(mask.sum()); membrane_norms.append(float(np.linalg.norm(membrane[i])))
                if count:
                    signs=np.sign(membrane[i,mask]).astype(np.int8)
                    prev=last_sign[i,mask]; rep=prev!=0
                    repeated += int(rep.sum()); reversals += int(np.sum(rep & (prev!=signs)))
                    last_sign[i,mask]=signs
                    agg[mask] += jump*signs.astype(np.float32)
                    membrane[i,mask]=0
                    events += count; bits=count*event_bits; payload+=bits; packetized+=bits+64; messages+=1
            w += agg
        else:
            raise ValueError(method)

        if rnd==1 or rnd%config.eval_stride==0 or rnd==config.rounds:
            tr,_,_,_,_,_=predictive_metrics(w,federation.X_train_eval,federation.y_train_eval,layout=layout,regularization=config.regularization)
            _,ce,acc,macro,worst,_=predictive_metrics(w,federation.X_test,federation.y_test,layout=layout,regularization=config.regularization)
            history.append({"round":rnd,"train_objective":tr,"test_ce":ce,"test_accuracy":acc,"macro_accuracy":macro,"worst_class_accuracy":worst,"payload_bits":payload})

    hist=pd.DataFrame(history)
    tr,trce,_,_,_,_=predictive_metrics(w,federation.X_train_eval,federation.y_train_eval,layout=layout,regularization=config.regularization)
    _,ce,acc,macro,worst,per_class=predictive_metrics(w,federation.X_test,federation.y_test,layout=layout,regularization=config.regularization)
    result={
        "method":method,"dimension":d,"rounds":config.rounds,"local_steps":config.local_steps,"local_lr":config.local_lr,
        "final_train_objective":tr,"final_train_ce":trce,"final_test_ce":ce,"final_test_accuracy":acc,"final_macro_accuracy":macro,
        "final_worst_class_accuracy":worst,"whole_train_objective":float(hist.train_objective.mean()),"payload_bits":int(payload),
        "packetized_bits":int(packetized),"messages":messages,"coordinate_events":events,"events_per_message":events/messages if messages else 0,
        "mean_delta_norm":float(np.mean(delta_norms)),"final_membrane_norm":float(np.linalg.norm(membrane)) if method=="event_fedavg" else 0,
        "sign_reversal_rate":reversals/repeated if repeated else 0,"rho":config.rho,"threshold":config.threshold,"jump0":config.jump0,"jump_exponent":config.jump_exponent,
        "topk_fraction":config.topk_fraction,
    }
    for c,a in enumerate(per_class): result[f"class_{c}_accuracy"]=float(a)
    return result
