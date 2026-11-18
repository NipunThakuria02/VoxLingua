"""Plots render actual report arrays; no example values appear in production charts."""
import numpy as np
import plotly.graph_objects as go

COLORS=['#C8D2C2','#97A9BA','#C5B692','#98989A','#81968C','#B7A5A0','#646B70']


def style(fig,height=350):
    fig.update_layout(height=height,paper_bgcolor='#090909',plot_bgcolor='#090909',font={'family':'Montserrat, sans-serif','color':'#ADADAD','size':11},
                      margin={'l':25,'r':20,'t':35,'b':30},legend={'orientation':'h','y':1.12},hoverlabel={'bgcolor':'#232323','font_size':11})
    fig.update_xaxes(gridcolor='#252525',zerolinecolor='#333',showline=True,linecolor='#333')
    fig.update_yaxes(gridcolor='#252525',zerolinecolor='#333',showline=True,linecolor='#333')
    return fig


def waveform(plots):
    fig=go.Figure(go.Scatter(x=plots['waveform_time_s'],y=plots['waveform_amplitude'],line={'color':'#B9C7B3','width':1},name='Original waveform'))
    fig.update_layout(xaxis_title='Time (s)',yaxis_title='Amplitude (full-scale units)')
    return style(fig,210)


def spectrogram(plots,three_d=False):
    z=np.asarray(plots['spectrogram_dbfs']);freq=np.asarray(plots['spectrogram_frequency_hz']);time=np.asarray(plots['spectrogram_time_s'])
    colors=[[0,'#0C1011'],[.35,'#26343C'],[.65,'#718896'],[.85,'#B1C4C8'],[1,'#EBEDE7']]
    if three_d:
        # Decimate actual bins/frames for interactive laptop performance, never synthesize data.
        fi=max(1,len(freq)//120);ti=max(1,len(time)//160)
        fig=go.Figure(go.Surface(x=time[::ti],y=freq[::fi],z=z[::fi,::ti],colorscale=colors,colorbar={'title':'dBFS'},hovertemplate='Time %{x:.2f}s<br>Frequency %{y:.0f}Hz<br>Level %{z:.1f}dBFS<extra></extra>'))
        fig.update_layout(scene={'xaxis_title':'Time (s)','yaxis_title':'Frequency (Hz)','zaxis_title':'Power (dBFS)',
                               'bgcolor':'#090909','camera':{'eye':{'x':1.6,'y':-1.6,'z':.85}}})
    else:
        fig=go.Figure(go.Heatmap(x=time,y=freq,z=z,colorscale=colors,colorbar={'title':'dBFS'},hovertemplate='%{x:.2f}s · %{y:.0f}Hz · %{z:.1f}dBFS<extra></extra>'))
        fig.update_layout(xaxis_title='Time (s)',yaxis_title='Frequency (Hz)')
    return style(fig,430)


def class_scores(scores):
    fig=go.Figure(go.Bar(x=list(scores.values()),y=[n.replace('_',' ') for n in scores],orientation='h',marker_color='#AABBA3',hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
    fig.update_layout(xaxis_title='Model score · uncalibrated',xaxis_range=[0,1],yaxis={'autorange':'reversed'})
    return style(fig,285)


def comparison(models,metrics):
    fig=go.Figure()
    for i,metric in enumerate(metrics):
        fig.add_bar(name=metric.replace('_',' ').title(),x=[n.replace('_',' ').title() for n in models],y=[models[n][metric] for n in models],marker_color=COLORS[i])
    fig.update_layout(barmode='group',yaxis_range=[0,1],yaxis_title='Held-out metric')
    return style(fig,390)


def confusion(matrix,labels):
    fig=go.Figure(go.Heatmap(z=matrix,x=labels,y=labels,colorscale=[[0,'#121815'],[1,'#BCD0B0']],text=matrix,texttemplate='%{text}',showscale=False))
    fig.update_layout(xaxis_title='Predicted',yaxis_title='Actual')
    return style(fig,380)
