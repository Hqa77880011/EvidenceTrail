import numpy as np
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Linear(c.get('input_shape',[8])[-1],32),nn.SiLU(),nn.Linear(32,c.get('classes',3)))
    def forward(self,b):
        logits=self.net(b['x'].float());return {'probability':logits.softmax(-1),'logits':logits}
    def objective(self,b,p):return F.cross_entropy(p['logits'],b['y'].long())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,8)).astype('float32');return {'x':x,'y':x[:,:3].argmax(1),'group':np.arange(n)}


def session(problem,variant='full'):
    from scipy.special import softmax
    backend=np.asarray(problem['backend'],float);logits=backend.copy();posterior=softmax(logits);queries=problem['queries'];used=set();accepted={};trail=[];submitted=None;cost=0.;clock=0.
    entropy=lambda p:float(-(p[p>0]*np.log(p[p>0])).sum())
    def feasible(q):
        if not q.get('available',True) or q['id'] in used:return False
        live=np.flatnonzero(posterior>=problem.get('live_threshold',.05))
        return variant=='no_topology' or any(q['stage'] in problem['neighborhoods'][str(int(h))] for h in live)
    def contribution(q,outcome):
        likelihood=np.asarray(q['likelihood'])[outcome];other=(likelihood.sum()-likelihood)/max(len(likelihood)-1,1)
        return q.get('reliability',1)*np.log(np.maximum(likelihood,1e-10)/np.maximum(other,1e-10))
    def utility(q):
        likelihood=np.asarray(q['likelihood']);predictive=likelihood@posterior;expected=0.
        for outcome,probability in enumerate(predictive):expected+=probability*entropy(softmax(logits+contribution(q,outcome)))
        gain=entropy(posterior)-expected
        if variant in {'baseline','no_voi'}:gain=0.
        return gain-.2*q['cost']-.15*q['latency']-.1*q.get('risk',0)
    for command in problem['commands']:
        affordable=[q for q in queries if feasible(q) and cost+q['cost']<=problem['budget']]
        if not affordable:break
        proposed=max(affordable,key=utility);identifier=command.get('query',proposed['id']);matches=[q for q in affordable if q['id']==identifier]
        if not matches:raise ValueError('Command requests unavailable, repeated, out-of-topology, or unaffordable evidence')
        q=matches[0];before=posterior.copy();outcome=problem['observations'][identifier];signed=contribution(q,outcome);edit=command.get('edit','accept')
        if edit=='accept':accepted[identifier]=signed
        elif edit=='reject':accepted.pop(identifier,None)
        else:raise ValueError('Evidence edit must be accept or reject')
        logits=backend+sum(accepted.values(),start=np.zeros_like(backend));posterior=softmax(logits);used.add(identifier);cost+=q['cost'];clock+=q['latency'];leading=int(posterior.argmax())
        negative=[name for name,value in accepted.items() if value[leading]<0];counter=min(negative,key=lambda name:accepted[name][leading]) if negative else 'none_observed'
        record={'timestamp':clock,'actor':command['actor'],'case':problem['case'],'model_version':problem['model_version'],'evidence':identifier,'before':before.tolist(),'after':posterior.tolist(),'query':identifier,'edit':edit,'rationale':command['rationale'],'counterevidence':counter,'signed_contribution':signed.tolist(),'proposed':proposed['id']}
        trail.append(record)
        if command.get('escalate'):submitted='escalated';break
        if command.get('confirm'):
            if posterior.max()<=.8 and not command.get('override_rationale'):raise ValueError('Low-confidence disposition requires an explicit override rationale')
            submitted=leading;record['disposition']=leading;record['override_rationale']=command.get('override_rationale');break
    required=['timestamp','actor','case','model_version','evidence','before','after','query','edit','rationale','counterevidence']
    completeness=sum(sum(key in event and event[key] is not None and event[key]!='' for key in required) for event in trail)/max(len(trail)*len(required),1)
    reconstructed={};replay_error=0.
    for event in trail:
        if event['edit']=='accept':reconstructed[event['evidence']]=np.asarray(event['signed_contribution'])
        else:reconstructed.pop(event['evidence'],None)
        replayed=softmax(backend+sum(reconstructed.values(),start=np.zeros_like(backend)))
        replay_error=max(replay_error,float(np.max(abs(replayed-np.asarray(event['after'])))))
    result={'replay_max_error':replay_error,'posterior':posterior.tolist(),'disposition':submitted,'queries':len(used),'query_cost':cost,'investigation_time':clock,'audit_completeness':completeness,'trail':trail}
    if 'true_hypothesis' in problem:result['diagnostic_correct']=submitted==problem['true_hypothesis']
    return result


def execute(args):
    import json
    from pathlib import Path
    result=session(json.loads(Path(args.input).read_text()),args.variant);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'events.jsonl').open('a',encoding='utf-8') as handle:
        for event in result.pop('trail'):handle.write(json.dumps(event)+'\n')
    (out/'session.json').write_text(json.dumps(result,indent=2))
