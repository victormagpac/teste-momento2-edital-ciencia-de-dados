import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .prepare import slugify

colors = ["#00853B", "#FEBD01"]
sns.set_theme(style="whitegrid")
sns.set_palette(sns.color_palette(colors))

def group_graph(df, list_vars):
    query_data = (
        df
        .groupby(list_vars)
        .aggregate('count')
        .reset_index()
    )
    query_data = (
        query_data.assign(Quantidade = query_data.paciente_id)
        .sort_values(by='Quantidade', ascending=False)
    )
    query_data = query_data
    x_order = list(query_data[list_vars[2]].unique())
    no_info = 'Não informado'
    if x_order.count(no_info):
        x_order.remove(no_info)
        x_order.append(no_info)
    catp = sns.catplot(x=list_vars[1], y="Quantidade", hue=list_vars[0], col=list_vars[2], data=query_data, kind="bar", height=4, aspect=.7, col_order=x_order).set_titles("{col_name}")
    for x, y, z in [ax[0] for ax in catp.facet_data()]:
        ax = catp.facet_axis(x, y, z)
        for c in ax.containers:
            labels = [f'{(v.get_height() / len(df) * 100):.1f} %' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=9)
    plt.suptitle(f"Distribuição de {list_vars[0]} por {'IDH' if list_vars[2] == 'idh_label' else list_vars[2].replace('_', ' ').capitalize()}\n",  y=1.1, fontweight='bold')
    plt.show()


def cat_graph(df, var_name, axe):
    query_data = df.groupby(var_name).aggregate('count').reset_index()
    ax = sns.barplot(y=var_name, x='paciente_id', data=query_data, color=colors[0], ax=axe)
    labels = [f'{(v.get_width() / len(df) * 100):.1f} %' for v in ax.containers[0]]
    graph = ax.bar_label(ax.containers[0], labels=labels, label_type='edge', fontsize=12)
    ax.set(
        xlabel='Quantidade',
        title=f"{var_name}"
    )
    sns.despine(left=True, bottom=False)
    return ax

def histogram(df, var_name, axe=None):
    ax = sns.histplot(data=df, x=var_name, alpha=1, color=colors[0], ax=axe)
    labels = [f'{(v.get_height() / len(df) * 100):.1f} %' for v in ax.containers[0]]
    graph = ax.bar_label(ax.containers[0], labels=labels, label_type='edge', fontsize=8)
    ax.xaxis.grid(False)
    ax.set(
        ylabel='Quantidade',
        title=f"{var_name}"
    )
    sns.despine(left=True, bottom=False)
    return ax

def correlation(df, title, annot=False):
# corr = df[df_dict.query("Grupo == 'Sintoma'").Coluna.to_list()].replace(["Não", "Sim"], [0, 1]).corr()
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 6))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=annot, fmt=".2f")
    plt.suptitle(f"Correlação ({title})", fontweight='bold')
    plt.show()

def correlation_between(df, var_list_1, var_list_2, title, annot=False):
    corr = (
        df.replace(["Não", "Sim"], [0, 1])
        .corr()
        .query('index in @var_list_1')[var_list_2]
    )
    # corr = df[df_dict.query("Grupo == 'Sintoma'").Coluna.to_list()].replace(["Não", "Sim"], [0, 1]).corr()
    # corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 6))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=annot, fmt=".2f")
    plt.suptitle(f"Correlação ({title})", fontweight='bold')
    plt.show()

def fortaleza_cases(df, counties):
    mapping = {
        bairro: slugify(bairro)
        for bairro in df.bairro.unique()
    }
    
    query_data = df.assign(bairro = df.bairro.replace(mapping)).groupby('bairro').aggregate('count').reset_index()
    query_data = (
        query_data
        .assign(quantidade = query_data.paciente_id)
        .query("bairro != 'NAO INFORMADO'")
        [['bairro', 'quantidade']]
    )
    bairros = [location['properties']['NOME'] for location in counties['features']]
    query_data = pd.concat([
        pd.DataFrame({
            'bairro': [bairro for bairro in set(bairros) - set(query_data.bairro.to_list())],
            'quantidade': [0 for bairro in set(bairros) - set(query_data.bairro.to_list())],
        }),
        query_data
    ], axis=0)
    
    fig = go.Figure(data=go.Choropleth(
        locations=query_data['bairro'],
        z=query_data['quantidade'].astype(float),
        geojson=counties,
        colorscale='Reds',
        autocolorscale=False,
        featureidkey="properties.NOME",
        # text= 'Casos: ' + query_data['quantidade'].astype(str),
        # locationmode="geojson-id",
        marker_line_color='white',
        colorbar_title="Quantidade",
    ))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title_text='Distribuição de casos em Fortaleza<br>(Gráfico interativo)',
        height=500,
        width=900,
        geo = dict(
            showlakes=True, # lakes
            lakecolor='rgb(255, 255, 255)'),
    )
    fig.show()

def results_plot(df, title):
    catp = (
        sns
        .catplot(x='model', y='mean', col='metric', data=df, kind="bar")#, col_wrap=2)
        .set_titles("{col_name}")
        .set_ylabels("Média")
    )
    for x, y, z in [ax[0] for ax in catp.facet_data()]:
        ax = catp.facet_axis(x, y, z)
        for c in ax.containers:
            labels = [f'{v.get_height():.2f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=12)
    plt.suptitle(title,  y=1.1, fontweight='bold')
    plt.show()