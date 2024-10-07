import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import pandas as pd
import spacy
# from spacy import displacy
import numpy as np
from pyvis.network import Network
from networkx.algorithms.community.centrality import girvan_newman
import re
import matplotlib.pyplot as plt
import os
import community as community_louvain
import python.lib.utils.functions as fn
import matplotlib.pyplot as plt
from node2vec import Node2Vec


class network_analysis:
    def load_books (self):
        return( [b for b in os.scandir('data') if '.txt' in b.name])

    def filter_entity(self, ent_list, character_df):
        return[ent for ent in ent_list
               if ent in list(character_df.character)
               or ent in list(character_df.character_firstname)
               ]


    def load_characters(self):
        print("Characters loading")
        character_df = pd.read_csv('../data/characters.csv')
        # remove all empty () and text within () and added a first name column in the same data frame
        character_df['character'] = character_df['character'].apply(lambda x: re.sub("[\(].*?[\)]", "", x))
        character_df['character_firstname'] = character_df['character'].apply(lambda x: x.split(' ', 1)[0])
        print("Characters load completed.")
        return(character_df)

    def process_sent_entities(self, character_df, sent_entity_df):
        # #filter out non-character entities e.g '1000' apply the function
        print("Processing entities that are not characters..")
        sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(
            lambda x: na.filter_entity(x, character_df))

        # # Filter out sentences that don't have any character entities
        sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].map(len) > 0]

        # take only first name of character
        sent_entity_df_filtered['character_entities'] = sent_entity_df_filtered['character_entities'].apply(
            lambda x: [item.split()[0] for item in x])
        print("Process entity characters completed.")
        return (sent_entity_df_filtered)

    def create_visual_graph(self, relationships_df):

        # create graph from pandas dataframe
        G = nx.from_pandas_edgelist(relationships_df,
                                    source="source",
                                    target="target",
                                    edge_attr="value",
                                    create_using=nx.Graph())

        colors = ['#008B8B', 'b', 'orange', 'y', 'c', 'DeepPink', '#838B8B', 'purple', 'olive', '#A0CBE2',
                  '#4EEE94'] * 50
        colors = colors[0:len(G.nodes())]
        na.draw_networkx(G, 1,colors)

        louvain_com(G,3)

        colors = ['#008B8B', 'b', 'orange', 'y', 'c', 'DeepPink', '#838B8B', 'purple', 'olive', '#A0CBE2',
                  '#4EEE94'] * 50
        colors = colors[0:len(G.nodes())]
        na.draw_networkx(G, 2,colors)
        self.newman_com(G)

    def draw_networkx(self, G, fig, colors):
        plt.figure(fig, figsize=(15, 10), dpi=400)

        nx.draw_networkx(G, pos=nx.spring_layout(G),  # Use the spring layout for positioning nodes
                         node_color=colors,  # Specify node colors
                         edge_color=colors,  # Specify edge colors
                         font_color='white',  # Font color for node labels
                         node_size=50,  # Size of the nodes
                         font_size=2,  # Font size for node labels
                         alpha=0.95,  # Transparency level
                         width=0.3,  # Width of the edges
                         font_weight=9

                         )
        plt.axis('off')
        plt.show()

    def newman_com(self, G):
        communities = girvan_newman(G)
        node_groups = []
        for com in next(communities):
            node_groups.append(list(com))

        colors = []
        colors = ["blue" if node in node_groups[0] else "green" for node in G]
        na.draw_networkx(G, 4, colors)

    def louvain_com(self, G):
        communities = community_louvain.best_partition(G)
        nx.set_node_attributes(G, communities, 'group')
        #more to come.


if __name__ == "__main__":

    na = network_analysis()

    all_books = na.load_books()
    book = all_books[1]

    print("Extracting entities from each sentence...")
    book_doc = fn.ner(book)
    sent_entity_df = fn.get_ne_list_per_sentence(book_doc)

    print("Extract entities completed.")
    character_df = na.load_characters()
    sent_entity_df_filtered = na.process_sent_entities(character_df, sent_entity_df)
    # create relationships
    print("Creating relationship in progress...")
    relationships_df = fn.create_relationships(sent_entity_df_filtered, window_size=5)
    print("Creating relationship completed.")

    na.create_visual_graph(relationships_df)


