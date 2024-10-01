from gprofiler import GProfiler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def go_enrichment(gene_list, organism='hsapiens'):
    """ Perform GO enrichment analysis using g:Profiler focusing on Cellular Components. """
    gp = GProfiler(return_dataframe=True)
    # Run the enrichment analysis
    results = gp.profile(organism=organism, query=gene_list, sources=['GO:CC'])
    
    # Filter results for significance and output
    significant_results = results[results['p_value'] < 0.05]  # Adjust p-value threshold if necessary
    return significant_results

# Example gene list
lost_preys = pd.read_csv('a.csv').columns.to_list()

# Perform the enrichment
enrichment_results = go_enrichment(lost_preys)



def plot_results(results):
    # Sort results by p-value and select the top terms for plotting
    results = results.sort_values(by='p_value').head(5)
    plt.figure(figsize=(8, 6))
    plt.barh(results['name'], -np.log10(results['p_value']))
    plt.xlabel('-Log10(p-value)', fontsize=16)
    # plt.ylabel('GO Terms')
    # plt.title('Top 5 Significant GO CC Terms')
    plt.xticks(fontsize=16)  
    plt.yticks(fontsize=16) 
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/lost_preys_enriched_terms.png', dpi=300)
    plt.savefig('plots/lost_preys_enriched_terms.svg', dpi=300)
    plt.savefig('plots/lost_preys_enriched_terms.pdf', dpi=300)


plot_results(enrichment_results)



