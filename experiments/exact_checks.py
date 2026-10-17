"""Exact finite models for the OPD occupancy note. CPU only, no sampling error.

Outputs: summary.json (asserted identities/counterexamples), curves.csv.
Every state/action distribution is explicitly enumerated. These constructed
models test mathematical claims; they do not measure a trained language model.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z)))


def logit(p):
    return np.log(p) - np.log1p(-p)


def kl(p, q):
    return p * np.log(p / q) + (1-p) * np.log((1-p)/(1-q))


def softmax(z):
    v = np.exp(z - np.max(z))
    return v/v.sum()


def finite_diff(f, z, h=1e-5):
    eye = np.eye(len(z))
    return np.array([(f(z+h*v)-f(z-h*v))/(2*h) for v in eye])


def categorical_checks():
    p = np.array([.05, .3, .65])
    q = np.array([.7, .2, .1])
    mu = np.array([.8, .1, .1])
    score = np.eye(3)-p[None,:]
    e = np.log(p/q)
    local = (p[:,None]*e[:,None]*score).sum(0)
    offline = (mu[:,None]*e[:,None]*score).sum(0)
    corrected = (mu[:,None]*(p/mu)[:,None]*e[:,None]*score).sum(0)
    fd_kl = finite_diff(lambda z: np.sum(softmax(z)*np.log(softmax(z)/q)), np.log(p))
    fd_sq = finite_diff(lambda z: .5*np.sum(mu*np.log(softmax(z)/q)**2), np.log(p))
    errors = {
        'on_policy_vs_kl': float(np.max(np.abs(local-fd_kl))),
        'offline_vs_squared_log': float(np.max(np.abs(offline-fd_sq))),
        'importance_corrected_vs_kl': float(np.max(np.abs(corrected-local))),
    }
    assert max(errors.values()) < 1e-8, errors
    return dict(errors=errors, on_policy_gradient=local.tolist(), offline_gradient=offline.tolist(),
                offline_bias_norm=float(np.linalg.norm(offline-local)))


def tree_checks():
    # State 0 chooses L (action 1) or R; states L and R choose success (action 1).
    q = np.array([.9, .999, .9])
    p = np.array([.8, .001, .9])
    z = logit(p)
    weights = np.array([1., p[0], 1-p[0]])
    local = weights*p*(1-p)*(logit(p)-logit(q))
    missing = np.array([p[0]*(1-p[0])*(kl(p[1],q[1])-kl(p[2],q[2])),0.,0.])
    def seq(z):
        a,b,c = sigmoid(z)
        return kl(a,q[0])+a*kl(b,q[1])+(1-a)*kl(c,q[2])
    def fixed(z):
        return np.sum(weights*kl(sigmoid(z),q))
    def accuracy(z):
        a,b,c = sigmoid(z)
        return a*b+(1-a)*c
    full = finite_diff(seq,z)
    assert np.max(np.abs(full-local-missing)) < 1e-8
    eta=.01
    new=z-eta*local
    assert seq(new)>seq(z) and fixed(new)<fixed(z) and accuracy(new)<accuracy(z)
    result=dict(p=p.tolist(),q=q.tolist(),local_gradient=local.tolist(),
                full_gradient=full.tolist(),missing_gradient=missing.tolist(),
                dot_full_local=float(full@local),step=eta,
                sequence_kl_before=float(seq(z)),sequence_kl_after=float(seq(new)),
                frozen_prefix_kl_before=float(fixed(z)),frozen_prefix_kl_after=float(fixed(new)),
                accuracy_before=float(accuracy(z)),accuracy_after=float(accuracy(new)),
                teacher_accuracy=float(q[0]*q[1]+(1-q[0])*q[2]))
    rows=[]
    for i in range(101):
        rows.append(dict(experiment='tree',arm='local',step=i,x=i,y=float(seq(z)),metric='sequence_kl'))
        rows.append(dict(experiment='tree',arm='local',step=i,x=i,y=float(accuracy(z)),metric='accuracy'))
        p=sigmoid(z)
        z-=eta*np.array([1.,p[0],1-p[0]])*p*(1-p)*(logit(p)-logit(q))
    return result,rows


def projection_checks():
    # Both states share exactly one logit, but their teachers prefer opposites.
    q = np.array([.9,.1])
    rows=[]
    finals={}
    starts={}
    for w in [.1,.5,.9]:
        target=w*logit(q[0])+(1-w)*logit(q[1])
        z=0.
        for i in range(1001):
            p=float(sigmoid(z))
            rows.append(dict(experiment='projection',arm=f'w={w}',step=i,x=i,y=p,metric='p'))
            if i<1000:
                z-=.5*p*(1-p)*(z-target)
        assert abs(z-target)<1e-10
        finals[str(w)]=dict(p=p,analytic_p=float(sigmoid(target)),
                           evaluation_accuracy=.2+.6*p,
                           irreducible_kl=float(w*kl(p,.9)+(1-w)*kl(p,.1)))
        starts[w]=z
    common=.7*logit(.9)+.3*logit(.1)
    handoff={}
    for w,z in starts.items():
        for i in range(301):
            p=float(sigmoid(z))
            rows.append(dict(experiment='handoff',arm=f'w={w}',step=i,x=i,y=p,metric='p'))
            if i<300:
                z-=.5*p*(1-p)*(z-common)
        handoff[str(w)]=p
        assert abs(p-float(sigmoid(common)))<1e-10
    # In a realizable tabular model, state weight changes speed but not the limit.
    for w in [.1,.5,.9]:
        z=np.zeros(2)
        weights=np.array([w,1-w])
        for _ in range(1000):
            z=(1-.2*weights)*z+.2*weights*logit(q)
        assert np.max(np.abs(sigmoid(z)-q))<1e-8
    return dict(finals=finals,handoff=handoff,handoff_target=float(sigmoid(common)),
                tabular_limit=q.tolist()),rows


def variance_checks():
    rows=[]
    summary=[]
    for p in np.logspace(-7,-2,61):
        q=.9
        probs=np.array([p,1-p])
        values=np.array([(1-p)*np.log(p/q),-p*np.log((1-p)/(1-q))])
        mean=float(probs@values)
        var=float(probs@((values-mean)**2))
        ratio=var/mean**2
        rows.extend([dict(experiment='variance',arm='sampled',step=0,x=p,y=var,metric='absolute_variance'),
                     dict(experiment='variance',arm='sampled',step=0,x=p,y=ratio,metric='relative_variance')])
        if len(summary)==0 or abs(np.log10(p)-round(np.log10(p)))<1e-10:
            summary.append(dict(p=float(p),mean=mean,variance=var,relative_variance=ratio))
    assert summary[0]['variance'] < summary[-1]['variance']
    assert summary[0]['relative_variance'] > summary[-1]['relative_variance']
    return summary,rows


def mixing_checks():
    # Enumerate two binary autoregressive policies with prefix-dependent logits.
    p=np.array([.2,.7,.4]); q=np.array([.8,.3,.9]); lam=.5
    def path(v):
        a,b,c=v
        return np.array([a*b,a*(1-b),(1-a)*c,(1-a)*(1-c)])
    P,Q=path(p),path(q)
    token_fusion=sigmoid((1-lam)*logit(p)+lam*logit(q))
    fused=path(token_fusion)
    arithmetic=path((1-lam)*p+lam*q)
    batch=(1-lam)*P+lam*Q
    sequence_geometric=np.sqrt(P*Q); sequence_geometric/=sequence_geometric.sum()
    assert not np.allclose(fused,batch)
    assert not np.allclose(fused,sequence_geometric)
    assert not np.allclose(arithmetic,batch)
    assert np.max(P/batch)<=1/(1-lam)+1e-12
    assert np.max(Q/batch)<=1/lam+1e-12
    # Teacher suffix cannot erase first-token mismatch.
    st=np.array([p[0]*q[1],p[0]*(1-q[1]),(1-p[0])*q[2],(1-p[0])*(1-q[2])])
    kl_st=float(np.sum(st*np.log(st/Q)))
    assert abs(kl_st-float(kl(p[0],q[0])))<1e-12
    return dict(P=P.tolist(),Q=Q.tolist(),batch=batch.tolist(),fusion=fused.tolist(),
                token_arithmetic=arithmetic.tolist(),sequence_geometric=sequence_geometric.tolist(),
                student_then_teacher=st.tolist(),student_then_teacher_kl=kl_st,
                root_kl=float(kl(p[0],q[0])))


def further_identities():
    p=np.array([.5,.001,.499]); q=np.array([.6,.39,.01])
    direction=np.array([-.01,.1,-.09])
    slope=float(direction@np.log(p/q))
    new=p+.001*direction
    assert slope<0 and np.sum(new*np.log(new/q))<np.sum(p*np.log(p/q))
    assert new[0]<p[0]<q[0]
    p=np.array([.05,.3,.65]); q=np.array([.7,.2,.1]); lam=.4
    r=np.log(q/p)
    def blend(l):
        return softmax(np.log(p)+l*r)
    m=blend(lam)
    variance=float(m@((r-m@r)**2))
    h=1e-5
    errors=[]
    for target,expected in [(p,lam*variance),(q,(lam-1)*variance)]:
        def distance(l):
            m=blend(l)
            return np.sum(m*np.log(m/target))
        derivative=(distance(lam+h)-distance(lam-h))/(2*h)
        errors.append(float(abs(derivative-expected)))
    assert max(errors)<1e-8
    eigenvalues=np.array([.01,.1,1.])
    coefficients=np.array([2.,1.,.5])
    def absorption(t):
        weights=eigenvalues*coefficients**2*np.exp(-2*eigenvalues*t)
        weights/=weights.sum()
        return float(2*weights@eigenvalues),weights
    rate,weights=absorption(3.)
    expected=-4*float(weights@((eigenvalues-weights@eigenvalues)**2))
    measured=(absorption(3.+h)[0]-absorption(3.-h)[0])/(2*h)
    assert abs(measured-expected)<1e-8 and expected<0
    return dict(kl_descent_accuracy_descent_slope=slope,
                fusion_distance_derivative_errors=errors,
                quadratic_absorption_rate=rate,
                quadratic_absorption_rate_derivative=expected,
                quadratic_derivative_error=float(abs(measured-expected)))


def main():
    results={'metadata':dict(device='cpu',dtype='float64',randomness='none; exact enumeration',
                             python=platform.python_version(),numpy=np.__version__,
                             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())}
    results['categorical']=categorical_checks()
    results['tree'],a=tree_checks()
    results['projection'],b=projection_checks()
    results['variance'],c=variance_checks()
    results['mixing']=mixing_checks()
    results['further_identities']=further_identities()
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    with (out/'curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['experiment','arm','step','x','y','metric'],lineterminator='\n')
        writer.writeheader(); writer.writerows(a+b+c)
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
