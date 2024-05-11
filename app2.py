## Import das bibliotecas

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import matplotlib as plt
import requests
import joblib
from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.tsa.stattools import adfuller

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utilsforecast.losses import rmse, mape, mae

st.set_page_config(layout='wide')

st.title('Dashboard Barril tipo Brent:')

### Coleta dados históricos do índice de referência até a data corrente

ticker = 'BZ=F'  # Brent Crude Oil Last Day Financ (BZ=F)
start_date = '2009-01-01' #Define a data de ínicio para importação dos dados

brent_data = yf.download(ticker, start_date) #Quando a biblioteca é chamada sem uma data final, carrega as cotações até a data corrente
brent = pd.DataFrame({ticker: brent_data['Close']})
brent.rename(columns={ticker: 'y'}, inplace=True)
brent.rename_axis('ds', inplace=True)

#brent_bkp = brent.to_csv('brent_bkp.csv', sep=';')

#st.write(brent.tail())

aba1, aba2, aba3, aba4, aba5 = st.tabs(['Análise dos Dados e Insights', 'Análise para preparação e modelagem ' , 'Avaliação dos modelos', 'Previsões', 'Extras'])

### Análise dos Dados - Página 1

# Graficos (evolução, boxplot, seasonal_decompose)

# Evolução - series

fig_evolucao = px.line(brent, x=brent.index, y='y', 
                 template='plotly_dark',)

fig_evolucao.update_layout(
    title='Evolução - Barril tipo Brent - 2009 a 2024',
    title_font_color='white',  # Defina a cor do título aqui
    xaxis_title='',  # Define o rótulo do eixo x
    yaxis_title='US$',  # Define o rótulo do eixo y
    xaxis_title_font=dict(color='white'),  # Define a cor do rótulo do eixo x
    yaxis_title_font=dict(color='white'),  # Define a cor do rótulo do eixo y
    
    legend=dict(
        title=dict(
            text='',
            font=dict(
                color='gray'  # Defina a cor do título da legenda aqui
            )
        )
    ),
    
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

fig_evolucao.update_traces(marker=dict(color='white'), selector=dict(type='line'))

## Boxplot

fig_box= px.box(brent, y='y', 
                 template='plotly_dark',)

fig_box.update_layout(
    title='Distribuição dados do Brent',
    title_font_color='white', 
    xaxis_title='',  
    yaxis_title='US$', 
    xaxis_title_font=dict(color='white'), 
    yaxis_title_font=dict(color='white'), 
    
    legend=dict(
        title=dict(
            text='',
            font=dict(
                color='white'  
            )
        )
    ),
    
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, categoryorder='total ascending')
    
)

fig_box.update_traces(marker=dict(color='royalblue'))

# seasonal decompose

def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str='Seasonal Decomposition'):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residuals'],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode='lines', name='Observed'),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode='lines', name='Trend'),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode='lines', name='Seasonal'),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode='lines', name='Residual'),
            row=4,
            col=1,
        )
        .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False, template='plotly_dark'
        )
    )

decomposition = seasonal_decompose(brent['y'], model='additive', period=253)
fig_decompose = plot_seasonal_decompose(decomposition, dates=brent.index)

with aba1:

    coluna1, = st.columns(1)

    with coluna1:
        """ 
            - Os dados para o estudo preliminar compreendem o histórico de 2009 até 2024. A principio esse período justifica-se para buscarmos entender as oscilações ciclicas dos cenários ao longo do tempo com volume de dados relevante. Observa-se no periodo eventos turbulentos no ambiente interno e externo (politica, macroeconomia, pandemia, guerras).
            - Para a captura dos dados, utilizamos a biblioteca 'yfinance' do Python que nos conecta de forma on-line à base de dados da cotação do Brent.
            - O preço do petróleo é altamente influenciado pelas decisões sobre controle da produção globla da OPEP (Organização dos Países Exportadores de Petróleo), que tem como missão declarada "coordenar e unificar as políticas de petróleo de seus países membros e garantir a estabilização dos mercados de petróleo, a fim de garantir um fornecimento eficiente, econômico e regular deste recurso aos consumidores, uma renda estável aos produtores e um retorno justo de capital para aqueles que investem na indústria petrolífera".
        """
        st.plotly_chart(fig_evolucao, use_container_width= True)

        """ 
            A partir dessa primeira visualização, identificamos uma grande oscilação ao longo do tempo. Podemos observar alguns aspectos relevantes ao longo da série histórica:
            
            - Crise Global de 2008: a partir de 2009 (inicio dos dados) já temos uma crescente até os maiores picos da década de 2010 (2011 e 2012), logo após a grande crise econômica global de 2008;
            - Desaquecimento econômico global, observado no ano de 2015, em especial em grandes economias como a China, com queda expressiva do preço em 34% com relação ao ano anterior (com a cotação em torno de 37,10 dólares);
            - Recorde de produção mundial em 2018, estabilizando o preço que estava em recuperação, com queda expressiva até 2020, onde temos o auge da pandemia;
            - Guerra na Ucrânia em 2022 com preço recorde do barril no mercado internacional, atingindo picos acima de 120 dólares (preço recorde 127,98 em mar/22). No momento atual (2024), temos uma crescente demanda da China devido ao crescimento industrial, impactos da oferta da Russia devido à possiveis ataques às suas refinarias pela Ucrânia (redução de até 7% na produção), além do conflito em Gaza sem perspectivas de cessar fogo, que traz um cenário de preocupação. 
        """
        """ 
            Na análise de distribuição, observamos uma mediana de cerca 75 dólares e Q3 concentrado da casa de 100 dólares. Já o menor valor foi observado em abr/20 no auge da pandemia:       
        """
        
        st.plotly_chart(fig_box, use_container_width= True)
        
        """
            Logo abaixo podemos analisar com mais detalhes os componentes da série, com a tendência, sazonalidade e ruidos observados, no qual evidenciamos alguns aspectos de impactos apresentados, como objetivo desse trabalho:
                            
        """
        st.plotly_chart(fig_decompose, use_container_width= True)
        """
            É possível observarmos uma alta sazonalidade na série, bem como uma variação na tendência com oscilações bem relevantes em momentos distintos.
                            
        """
       
## Testes para Preparação e modelagem - Pg 2

##Graficos

# Media Movel 90

ma = brent.rolling(90).mean()

fig_movel = go.Figure()

fig_movel.add_trace(go.Scatter(x=ma.index, y=ma['y'], 
                         mode='lines', name='movel', line=dict(color='red')))

fig_movel.add_trace(go.Scatter(x=brent.index, y=brent['y'], 
                         mode='lines', name='real', line=dict(color='royalblue')))

fig_movel.update_layout(
    title='Brent x Medial Movel - 90 dias',
    title_font_color=('white'),
    xaxis_title='',
    yaxis_title='US$',
    xaxis_title_font=dict(color='white'),
    yaxis_title_font=dict(color='white'),
    #paper_bgcolor='dark',
    #plot_bgcolor='dark',
    template='plotly_dark',
    legend=dict(
        title=dict(
            text='',
            font=dict(
                color='white'
            )
        ),
        x=0.8,  
        y=1.0
    ),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

# log da media movel  

brent_log = np.log(brent )
ma_log = brent_log.rolling(90).mean()

fig_ma_log= go.Figure()

fig_ma_log.add_trace(go.Scatter(x=ma_log.index, y=ma_log['y'],
                         mode='lines', name='movel', line=dict(color='red')))

fig_ma_log.add_trace(go.Scatter(x=brent_log.index, y=brent_log['y'], 
                         mode='lines', name='real', line=dict(color='royalblue')))



fig_ma_log.update_layout(
    title='Brent log x Log da Medial Movel - 90 dias',
    title_font_color=('white'),
    xaxis_title='',
    yaxis_title='log_preço_brent',
    xaxis_title_font=dict(color='white'),
    yaxis_title_font=dict(color='white'),
    #paper_bgcolor='gray',
    #plot_bgcolor='gray',
    template='plotly_dark',
    legend=dict(
        title=dict(
            text='',
            font=dict(
                color='white'
            )
        ),
        x=0.8,  
        y=1.0
    ),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

# transformada logaritmica média movel e desvio padrão

brent_s = (brent_log - ma_log).dropna()

ma_s = brent_s.rolling(90).mean()

std = brent_s.rolling(90).std()


fig_std = go.Figure()

fig_std.add_trace(go.Scatter(x=ma_s.index, y=ma_s['y'],
                         mode='lines', name='movel', line=dict(color='red')))

fig_std.add_trace(go.Scatter(x=brent_s.index, y=brent_s['y'], 
                         mode='lines', name='real', line=dict(color='royalblue')))

fig_std.add_trace(go.Scatter(x=std.index, y=std['y'], 
                         mode='lines', name='std', line=dict(color='green')))

fig_std.update_layout(
    title='Brent - transformada logaritmica com media e desvio padrao 90 dias)',
    title_font_color=('white'),
    xaxis_title='',
    yaxis_title='log_movel_brent',
    xaxis_title_font=dict(color='white'),
    yaxis_title_font=dict(color='white'),
    #paper_bgcolor='white',
    #plot_bgcolor='white',
    template='plotly_dark',
    legend=dict(
        title=dict(
            text='',
            font=dict(
                color='gray'
            )
        ),
        x=0.8,  
        y=1.0
    ),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

result = adfuller(brent)
p_value = result[1].round(4)

result_ma = adfuller(ma.dropna())
p_value_90 = result_ma[1].round(4)

result_log = adfuller(brent_log.dropna())
p_value_log = result_log[1].round(4)

result_log_std = adfuller(brent_s)
p_value_log_std = result_log_std[1].round(4)

with aba2:

    coluna1, = st.columns(1)

    with coluna1:
        st.header('Media Movel - 90 dias')
        st.write('P-valor media movel 90 dias:', p_value_90, use_container_width= True)
        """ A série com a aplicação da média móvel não é estacionária"""
        st.plotly_chart(fig_movel, use_container_width= True)
        st.header('Log da Media Movel')
        st.write('P-valor log media movel :', p_value_log, use_container_width= True)
        """ A série com a aplicação do log da média móvel também não é estacionária"""
        st.plotly_chart(fig_ma_log, use_container_width= True)
        st.header('Transformada Logaritmica com media movel e desvio padrão')
        st.write('P-valor log media movel e desvio padrão :', p_value_log_std, use_container_width= True)
        """ A série com a transformação logaritimica com média móvel e desvio padrão é estacionária"""
        st.plotly_chart(fig_std, use_container_width= True)

        """ 
        As análises acima são muito importantes para o correto entendimento do comportamento da série e definição do modelo e seus parâmetros. Dado o fato de termos uma série originalmente não estacionária e somente com o tempo como parâmetro, iremos utilizar uma abordagem com o método Boosting, um tipo de método ensemble capaz de aprimorar os novos preditores com base em seus antecessores (utiliza ajuste dos pesos a cada iteração ou ajuste do preditor aos erros residuais do seu antecessor). 
                
        """
    

# Previsão do modelo

brent.reset_index(inplace=True)
brent.insert(0, 'unique_id', '1')

limit_day = brent['ds'].max() - pd.Timedelta(hours=2880) # ultimos 120 dias

# Split do dataframe para pegar sempre os ultimos 120 dias para base de previsão (processo batch) - conforme atualização diária dos dados
limit_day_pred = brent[brent['ds'] > limit_day]

# Modelos - Gradient Boosting Regressor - set com lag1, lag2 e lag5

model = joblib.load('modelo/gbr_ml_v5.joblib')

forecast_gbr = model.predict( h=15, level=[90])

# Random Forest Regressor

model_rf = joblib.load('modelo/rf_ml_v5.joblib')

forecast_rf = model_rf.predict(h=15, level=[90, 95])

fig_prev = go.Figure()

# Adicionar linha com as previsões para a próxima semana
fig_prev.add_trace(go.Scatter(x=forecast_gbr['ds'], y=forecast_gbr['GradientBoostingRegressor'], mode='lines+markers', name='Predict Gradient Boostimg Regressor'))

fig_prev.add_trace(go.Scatter(x=brent['ds'].tail(90), y=brent['y'].tail(90), mode='lines+markers', name='Real (histórico)'))

fig_prev.add_trace(go.Scatter(x=forecast_rf['ds'].tail(90), y=forecast_rf['RandomForestRegressor'].tail(90), mode='lines+markers', name='Predict Random Forest Regressor'))

today = pd.Timestamp(datetime.today())

with aba3:
    coluna1, = st.columns(1)
    
    with coluna1:
        metrics_dict = {'Gradient Boosting Regressor': 3.851392009152309, 
                        'Random Forest': 5.211964494492221,
                        'LightGBM Regressor': 5.676050999396097 }

        # Criar o gráfico de barras
        fig_metrics = go.Figure(data=[go.Bar(x=list(metrics_dict.keys()), y=list(metrics_dict.values()))])
        #fig_metrics = go.Figure(data=[go.Bar(x=list(metrics_dict.keys()), y=list(metrics_dict.values()), text=list(metrics_dict.values()), textposition='auto')])
        # Atualizar o layout do gráfico
        fig_metrics.update_layout(
                        title='Métricas de avaliação dos modelos (Cross-Validation - RMSE)',
                        title_font_color=('white'),
                        xaxis_title='',
                        yaxis_title='RMSE',
                        xaxis_title_font=dict(color='white'),
                        yaxis_title_font=dict(color='white'),
                        #font=dict(  
                        #        size=22  
                        #),
                        #paper_bgcolor='white',
                        #plot_bgcolor='white',
                        template='plotly_dark',
                        legend=dict(
                            title=dict(
                                text='',
                                font=dict(
                                    color='gray'
                                )
                            ),
                            x=0.8,  
                            y=1.0
                        ),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False)
        )

# MAPE

        metrics_dict_mape = {'Gradient Boosting Regressor': 0.04095859208349655 , 
                             'Random Forest': 0.05814974812931474,
                             'LightGBM Regressor': 0.06204769930256675  }

        fig_metrics_mape = go.Figure(data=[go.Bar(x=list(metrics_dict_mape.keys()), y=list(metrics_dict_mape.values()))])
        fig_metrics_mape.update_layout(
                            title='Métricas de avaliação dos modelos (Cross-Validation - MAPE)',
                            title_font_color=('white'),
                            xaxis_title='',
                            yaxis_title='MAPE',
                            xaxis_title_font=dict(color='white'),
                            yaxis_title_font=dict(color='white'),
                            template='plotly_dark',
                            legend=dict(
                            title=dict(
                            text='',
                            font=dict(
                            color='gray'
                                    )
                            ),
                            x=0.8,  
                            y=1.0
                            ),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False)
        )

# Exibir o gráfico
        """
        - Gradient Boosting Regressor - é um método que adiciona sequencialmente preditores à um conjunto, cada um corrigindo seu antecessor e tenta ajustar o preditor novo aos erros residuais cometidos pelo preditor anterior.

        - LGBMRegressor - LightGBM é uma estrutura de aumento de gradiente que usa algoritmos de aprendizagem baseados em árvore e tem como vantagem a velocidade de treinamento, precisão, e capacidade de lidar com dados em grande escala.

        - RandomForestRegressor - modelo regressor florestal aleatório que ajusta uma série de árvores de decisão de classificação em várias subamostras do conjunto de dados e usa a média para melhorar a precisão preditiva e controlar o ajuste excessivo.

        Nos testes realizados por meio do cross validation (validação cruzada), identificamos a melhor performance utilizando o Gradient Boosting Regressor, seguido do Random Forest Regressor e LightGBM Regressor:
    
        """
        st.plotly_chart(fig_metrics, use_container_width=True)
        st.plotly_chart(fig_metrics_mape, use_container_width=True)
        #st.write(metrics_dict)
        """
        Selecionamos o Gradiente Boosting Regressor para trabalharmos com nossas estimativas e o Random Forest de forma comparativa, trazendo a seguir uma visão de ambos modelos e suas previsões.
    
        """
        

with aba4:
    coluna1, = st.columns(1)

    with coluna1:
        st.header('Gradient Boosting Regressor e Random Forest Regressor')
        st.plotly_chart(fig_prev, use_container_width=True)

        #st.write(forecast_gbr)
        #st.write(forecast_rf)
        
        st.header('Selecione uma data para previsão:')
        min_date = min(forecast_gbr['ds'])
        max_date = max(forecast_gbr['ds'])
        selected_date = st.date_input('Selecione uma data', min_value=min_date, max_value=max_date, value=min_date)

        selected_date = pd.to_datetime(selected_date)

        # Filtrando os dados para a data selecionada
        filtered_forecast = forecast_gbr[forecast_gbr['ds'] == selected_date]

        st.header('Previsão para a data selecionada com Gradient Boosting')

        if not filtered_forecast.empty:
            prediction = filtered_forecast.iloc[0]['GradientBoostingRegressor']
            if not pd.isnull(prediction):  # Verifica se a previsão não é um valor nulo
                big_number = round(prediction, 2)
                st.markdown(f'### Valor da Previsão:')
                st.markdown(f'## {big_number:.2f} US$')
            else:
                st.warning('Nenhuma previsão encontrada para a data selecionada.')
        else:
            st.warning('Nenhuma previsão encontrada para a data selecionada.')


        # Plotando um gráfico de linha para mostrar a tendência do valor do Brent Futuro desde a data atual até a data selecionada
        forecast_till_selected_date = forecast_gbr[(forecast_gbr['ds'] >= today) & (forecast_gbr['ds'] <= selected_date)]
        fig_trend = go.Figure(data=[go.Scatter(x=forecast_till_selected_date['ds'], y=forecast_till_selected_date['GradientBoostingRegressor'], mode='lines')])
        fig_trend.update_layout(title='Tendência do Valor do Brent Futuro',
                                xaxis_title='Data',
                                yaxis_title='Valor do Brent Futuro (US$)')
        st.plotly_chart(fig_trend, use_container_width=True)


        # Filtrando os dados para a data selecionada

        filtered_forecast_rf= forecast_rf[forecast_rf['ds'] == selected_date]

        st.header('Previsão para a data selecionada com Random Forest')

        if not filtered_forecast_rf.empty:
            prediction_rf = filtered_forecast_rf.iloc[0]['RandomForestRegressor']
            if not pd.isnull(prediction_rf):  # Verifica se a previsão não é um valor nulo
                big_number_rf = round(prediction_rf, 2)
                st.markdown(f'### Valor da Previsão:')
                st.markdown(f'## {big_number_rf:.2f} US$')
            else:
                st.warning('Nenhuma previsão encontrada para a data selecionada.')
        else:
            st.warning('Nenhuma previsão encontrada para a data selecionada.')

        
        # Plotando um gráfico de linha para mostrar a tendência do valor do Brent Futuro desde a data atual até a data selecionada
        forecast_till_selected_date_rf = forecast_rf[(forecast_rf['ds'] >= today) & (forecast_rf['ds'] <= selected_date)]
        fig_trend_rf = go.Figure(data=[go.Scatter(x=forecast_till_selected_date_rf['ds'], y=forecast_till_selected_date_rf['RandomForestRegressor'], mode='lines')])
        fig_trend_rf.update_layout(title='Tendência do Valor do Brent Futuro',
                                xaxis_title='Data',
                                yaxis_title='Valor do Brent Futuro (US$)')
        st.plotly_chart(fig_trend_rf, use_container_width=True)
        

with aba5:
    """
    Nessa página, você poderá escolher em separado as datas das previsões por estimador e realizar um comparativo entre os modelos de uma outra forma:

    """
    #st.header('Gradiente Boosting Regressor x Random Forest Regressor')
    #st.plotly_chart(fig_prev, use_container_width=True)

    coluna1, coluna2 = st.columns(2)

    with coluna1:
        st.header('Estimador - GB Regressor')
        min_date = min(forecast_gbr['ds'])
        max_date = max(forecast_gbr['ds'])
        selected_date_gb = st.date_input('Gradient Boosting Regressor', min_value=min_date, max_value=max_date, value=min_date)
        st.subheader('Previsão para a data selecionada com Gradient Boosting Regressor')
        selected_date_gb = pd.to_datetime(selected_date_gb)
        filtered_forecast_gb = forecast_gbr[forecast_gbr['ds'] == selected_date_gb]
        if not filtered_forecast_gb.empty:
            big_number_gb = filtered_forecast_gb.iloc[0]['GradientBoostingRegressor'].round(2)
            st.markdown(f'### Valor da Previsão:')
            st.markdown(f'## {big_number_gb} US$')
        else:
            st.warning('Nenhuma previsão encontrada para a data selecionada com Gradient Boosting.')

        # Plotando um gráfico de linha para mostrar a tendência do valor do Brent Futuro para Gradient Boosting
        forecast_till_selected_date_gb = forecast_gbr[(forecast_gbr['ds'] >= today) & (forecast_gbr['ds'] <= selected_date_gb)]
        fig_trend_gb = go.Figure(data=[go.Scatter(x=forecast_till_selected_date_gb['ds'], y=forecast_till_selected_date_gb['GradientBoostingRegressor'], mode='lines')])
        fig_trend_gb.update_layout(title='Tendência do Valor do Brent Futuro (Gradient Boosting)',
                                xaxis_title='Data',
                                yaxis_title='Valor do Brent Futuro (US$)')
        st.plotly_chart(fig_trend_gb, use_container_width=True)

    with coluna2:
        st.header('Estimador - RF Regressor')
        min_date = min(forecast_gbr['ds'])
        max_date = max(forecast_gbr['ds'])
        selected_date_rf = st.date_input('Random Forest Regressor', min_value=min_date, max_value=max_date, value=min_date)
        st.subheader('Previsão para a data selecionada com Random Forest Regressor')
        selected_date_rf = pd.to_datetime(selected_date_rf)
        filtered_forecast_rf = forecast_rf[forecast_rf['ds'] == selected_date_rf]
        if not filtered_forecast_rf.empty:
            big_number_rf = filtered_forecast_rf.iloc[0]['RandomForestRegressor'].round(2)
            st.markdown(f'### Valor da Previsão:')
            st.markdown(f'## {big_number_rf} US$')
        else:
            st.warning('Nenhuma previsão encontrada para a data selecionada com Random Forest.')

        # Plotando um gráfico de linha para mostrar a tendência do valor do Brent Futuro para Random Forest
        forecast_till_selected_date_rf = forecast_rf[(forecast_rf['ds'] >= today) & (forecast_rf['ds'] <= selected_date_rf)]
        fig_trend_rf = go.Figure(data=[go.Scatter(x=forecast_till_selected_date_rf['ds'], y=forecast_till_selected_date_rf['RandomForestRegressor'], mode='lines')])
        fig_trend_rf.update_layout(title='Tendência do Valor do Brent Futuro (Random Forest)',
                                xaxis_title='Data',
                                yaxis_title='Valor do Brent Futuro (US$)')
        st.plotly_chart(fig_trend_rf, use_container_width=True)


