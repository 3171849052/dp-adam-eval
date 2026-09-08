"""Seed traces and means; all figures written as PNG and PDF."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from expv1.common import METHODS, LAYERS
from expv1.validate_expv1 import cli, validate


def plot(c,runs,output):
    validate(c,runs,output)
    dest=output/'figures'
    dest.mkdir(parents=True,exist_ok=True)
    frames={name:pd.concat([pd.read_csv(runs/f'seed{s}'/m/f'{name}_metrics.csv')
        for s in c['seeds'] for m in METHODS],ignore_index=True) for name in ('train','layer','eigenbin')}
    colors=dict(zip(METHODS,['C0','C1','C2']))
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(dest/f'{name}.{ext}',dpi=160)
        plt.close(fig)
    def trace(ax,df,col,method,style='-',label=None,x='step'):
        df=df.dropna(subset=[col])
        for _,f in df.groupby('seed'):
            ax.plot(f[x],f[col],style,color=colors[method],alpha=.25,lw=.7)
        mean=df.groupby(x)[col].mean()
        ax.plot(mean.index,mean.values,style,color=colors[method],lw=2,label=label or method)
        ax.set_xlabel(x); ax.set_ylabel(col)
    for col in ('test_accuracy','test_loss','train_loss'):
        fig,ax=plt.subplots(figsize=(8,4))
        for method in METHODS:
            trace(ax,frames['train'].query('method==@method'),col,method)
        ax.legend();save(fig,col)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,method in zip(axes,METHODS[1:]):
        for col,style in [('relmse_noisy','--'),('relmse_filtered','-')]:
            trace(ax,frames['train'].query('method==@method'),col,method,style,col)
        ax.set_title(method);ax.legend()
    save(fig,'gradient_reconstruction')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,method in enumerate(METHODS[1:]):
        for col,style in [('cosine_noisy','--'),('cosine_filtered','-')]:
            trace(axes[0,j],frames['train'].query('method==@method'),col,method,style,col)
        trace(axes[1,j],frames['train'].query('method==@method'),'snr_gain_db',method)
        axes[0,j].legend();axes[0,j].set_title(method)
    save(fig,'direction_and_snr')
    eb=frames['eigenbin']
    steps=sorted(eb.step.unique()); selected=list(dict.fromkeys([steps[0],steps[len(steps)//2],steps[-1]]))
    for step in selected:
        fig,axes=plt.subplots(4,3,figsize=(13,13))
        for i,layer in enumerate(LAYERS):
            f=eb[(eb.step==step)&(eb.layer==layer)]
            for j,col in enumerate(('signal_power','empirical_snr','mse_reduction')):
                ax=axes[i,j]
                for seed,g in f.groupby('seed'):
                    ax.plot(g.log10_lambda_mean,g[col],'.-',alpha=.4,label=f'seed {seed}')
                g=f.groupby('bin')[['log10_lambda_mean',col]].mean()
                ax.plot(g.log10_lambda_mean,g[col],'k.-',lw=2,label='mean')
                ax.set(xlabel='log10(mean Fisher eigenvalue)',ylabel=col,title=f'{layer}; refresh {step}')
        axes[0,0].legend();save(fig,f'eigenmode_refresh_{int(step)}')
    fig,axes=plt.subplots(4,2,figsize=(13,13))
    for i,layer in enumerate(LAYERS):
        for method in METHODS[1:]:
            f=frames['layer'].query('layer==@layer and method==@method')
            trace(axes[i,0],f,'kappa',method)
            for col,style in [('H_q10',':'),('H_q50','-'),('H_q90','--')]:
                trace(axes[i,1],f,col,method,style,f'{method} {col}')
        axes[i,0].set_title(layer);axes[i,0].set_yscale('log');axes[i,1].set_ylim(0,1)
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=7)
    save(fig,'beta1_scale')
    print(f'Figures generated: {dest}')


if __name__=='__main__':
    plot(*cli())
