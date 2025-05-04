import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pyvis.network import Network
from scipy.stats import pearsonr, spearmanr

from .html_utils import modify_html_file


def plot_heatmap(matrix, labels):
    """
    Plots a heatmap using the given matrix and corresponding labels.

    Parameters:
    matrix (numpy.ndarray): 2D array to visualize as a heatmap.
    labels (list): Labels for the x and y axes.
    """

    max_abs_value = np.max(np.abs(matrix))

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', vmin=-max_abs_value, 
                vmax=max_abs_value, xticklabels=labels, yticklabels=labels, linewidths=0.5)
    plt.title('Interaction Values')
    plt.show()


class ShapInteractions:
    """
    A class to manage and analyze SHAP interaction values.

    Parameters:
    shap_interactions (numpy.ndarray): 3-dimensional array of SHAP interaction values.
    feature_values (numpy.ndarray): 2-dimensional array of feature values.
    feature_names (list): List of features names.
    compute_correlations (bool): If True, computes Pearson's and Spearman's correlation coefficients during class initialisation. 
        This may affect initialisation time, especially for large datasets.


    Attributes:
    shap_interactions (numpy.ndarray): Stored 3-dimensional array of SHAP interaction values.
    feature_values (numpy.ndarray): Stored 2-dimensional array of feature values.
    feature_names (list): Stored list of features names.
    interaction_trends (numpy.ndarray): Calculated interaction trends.
    trend_coefs (numpy.ndarray): Trend coefficients.
    pearsons (numpy.ndarray): Pearson's correlation coefficients between feature values and interaction trends.
    spearmans (numpy.ndarray): Spearman's correlation coefficients between feature values and interaction trends.
    """

    def __init__(self, shap_interactions=None, feature_values=None, feature_names=None, compute_correlations=False):

        if shap_interactions is None:
            raise Exception("No SHAP interaction values were provided.")
        
        if feature_values is None:
            raise Exception("No feature values were provided.")
        
        
        if not isinstance(shap_interactions, np.ndarray):
            raise TypeError("'shap_interactions' must be a numpy array.")
        
        if not isinstance(feature_values, np.ndarray):
            raise TypeError("'feature_values' must be a numpy array.")
        
        if shap_interactions.ndim != 3:
            raise ValueError("'shap_interactions' must be 3-dimensional.")
        
        if feature_values.ndim != 2:
            raise ValueError("'feature_values' must be 2-dimensional.")
        
        if feature_names is not None and len(feature_names) != feature_values.shape[1]:
            raise ValueError(
                "The length of 'feature_names' must match the number of features in 'feature_values'. "
                f"Expected {feature_values.shape[1]}, but got {len(feature_names)}."
            )
        
        if feature_names is None:
            self.feature_names = [f"feat_{nb}" for nb in range(1, feature_values.shape[1])]
        else:
            self.feature_names = feature_names

        self.shap_interactions = shap_interactions.copy()
        self.feature_values = feature_values.copy()

        m = shap_interactions.shape[1]
        n = shap_interactions.shape[0]

        interaction_trends = shap_interactions * np.sign(shap_interactions[:, np.arange(m), np.arange(m)].reshape(n, m, 1))

        self.interaction_trends = np.transpose(interaction_trends, (0, 2, 1))
        k = len(feature_names)
        self.pearsons, self.pval_pearsons = np.zeros((k,k)), np.zeros((k,k))
        self.spearmans, self.pval_spearmans = np.zeros((k,k)), np.zeros((k,k))

        if compute_correlations:
            self.compute_trend_coefs()


    def compute_trend_coefs(self):
        """
        Computes Pearson's and Spearman's correlation coefficients between feature values and interaction trends.
        """

        print(
            "Computing Pearson's and Spearman's correlation coefficients... This may take time for large datasets. "
            "If found, NaNs will be ignored."
        )

        for ind1 in range(self.interaction_trends.shape[1]):
            for ind2 in range(self.interaction_trends.shape[1]):
                y = self.interaction_trends[:, ind1, ind2].squeeze() if ind1 != ind2 else self.shap_interactions[:, ind1, ind2].squeeze()
                x = self.feature_values[:, ind1].flatten()
                
                valid_mask = ~np.isnan(x) & ~np.isnan(y)

                self.pearsons[ind1, ind2], self.pval_pearsons[ind1, ind2] = pearsonr(x[valid_mask], y[valid_mask])
                self.spearmans[ind1, ind2], self.pval_spearmans[ind1, ind2] = spearmanr(x[valid_mask], y[valid_mask])


    def get_interaction_matrix(self, mode='pos_neg', empty_diag=False, empty_top=False, heatmap=True):
        """
        Computes an interaction matrix based on specified mode.

        Parameters:
        mode (str): The computation mode for the interaction matrix. Options are 'avg', 'avg_abs', and 'pos_neg'.
        empty_diag (bool): If True, sets the diagonal elements of the matrix to zero.
        empty_top (bool): If True, retains only the lower triangular part of the matrix.
        heatmap (bool): If True, displays a heatmap of the interaction matrix.

        Returns:
        numpy.ndarray: The computed interaction matrix.
        """

        if mode not in ['avg', 'avg_abs', 'pos_neg']:
            raise ValueError("Invalid mode specified. Choose from 'avg', 'avg_abs', or 'pos_neg'.")

        if mode=='avg':
            interaction_matrix = np.nanmean(self.shap_interactions, axis=0)
        
        elif mode=='avg_abs':
            interaction_matrix = np.nanmean(np.abs(self.shap_interactions), axis=0)

        elif mode=='pos_neg':
            positive_mask = np.where(self.shap_interactions > 0, self.shap_interactions, 0)
            negative_mask = np.where(self.shap_interactions < 0, self.shap_interactions, 0)

            positive_mean = np.nanmean(positive_mask, axis=0)
            negative_mean = np.nanmean(negative_mask, axis=0)
            
            lower_triangular_1 = np.tril(positive_mean)
            upper_triangular_2 = np.triu(negative_mean, k=1)

            interaction_matrix = lower_triangular_1 + upper_triangular_2

            average_interaction_matrix = np.mean(self.shap_interactions, axis=0)

            np.fill_diagonal(interaction_matrix, np.diagonal(average_interaction_matrix))

        if empty_diag:
            np.fill_diagonal(interaction_matrix, 0)

        if empty_top:
            interaction_matrix = np.tril(interaction_matrix)

        if heatmap:
            plot_heatmap(interaction_matrix, self.feature_names)

        return interaction_matrix

    

    def create_graph(self, filename='interaction_graph.html', spearmans_threshold=0.3, max_arrows=200):
        """
        Builds the interaction graph and saves it to the specified HTML file.

        Parameters:
        filename (str): The name of the HTML file to which the graph will be saved.
        spearmans_threshold (float): The threshold used to determine whether an edge should be dashed or not. 
            Any edge with a Spearman's correlation coefficient lesser than this threshold will be dashed.
        """

        if np.all(self.spearmans==0) or np.all(self.pearsons==0):
            print("This step will be ignored next time (if already computed coefficients are found)")
            self.compute_trend_coefs()


        self.edges_data = []
        default_edges = []
        threshold = 0.2
        mean_shap_interactions = abs(self.shap_interactions).mean(axis=0)
        mean_shap_interactions_no_diag = mean_shap_interactions.copy()

        np.fill_diagonal(mean_shap_interactions_no_diag, float('-inf'))

        pairs = []

        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                value = mean_shap_interactions_no_diag[i][j]
                pairs.append((value, i, j))
        
        pairs.sort(reverse=True, key=lambda x: x[0])

        top_pairs = pairs[:max_arrows//2]
        min_val = top_pairs[-1][0]

        mean_shap_interactions_norm = (mean_shap_interactions - min_val) / (mean_shap_interactions_no_diag.max() - min_val)

        shadow_dict = {
            "enabled": True,
            "color": "rgba(0,0,0,0.5)",
            "size": 10,
            "x": 5,
            "y": 5
        }

        net = Network(notebook=True, cdn_resources='in_line', directed=True)

        diagonal_values = np.diag(mean_shap_interactions_norm)
        diag_min = diagonal_values.min()
        diag_max = diagonal_values.max()

        for i in range(len(self.feature_names)):
            if np.sign(self.spearmans[i,i]) != np.sign(self.pearsons[i,i]) or abs(self.spearmans[i,i])<spearmans_threshold:
                bg = 'black'
                bg_h = 'lightgrey'
            else:
                bg = '#fa3c62' if self.spearmans[i,i]>0 else '#3c8efa'
                bg_h = '#fa6180' if self.spearmans[i,i]>0 else '#90bcf5'

            pearson_val = self.pearsons[i, i]
            pearson_pval = self.pval_pearsons[i, i]

            spearman_val = self.spearmans[i, i]
            spearman_pval = self.pval_spearmans[i, i]

            main_title = (
                f"{self.feature_names[i]}\n"
                f"Pearson's r: {pearson_val:.2f} (pval {'< 0.0001' if pearson_pval < 0.0001 else '= ' + f'{pearson_pval:.4f}'})\n"
                f"Spearman's r: {spearman_val:.2f} (pval {'< 0.0001' if spearman_pval < 0.0001 else '= ' + f'{spearman_pval:.4f}'})\n"
                f"Mean absolute SHAP: {str(round(mean_shap_interactions[i,i],2))}"
            )

            val = mean_shap_interactions_norm[i][i]
            scaled = 2 + (val - diag_min) / (diag_max - diag_min) * (40 - 2) if diag_max != diag_min else 21

            net.add_node(
                self.feature_names[i],
                size=round(min(40, max(2, scaled))),
                color={
                    'background': bg,
                    'border': 'black',
                    'highlight': {
                        'background': bg_h,
                        'border': 'black'
                    }
                },
                shadow=shadow_dict,
                title=main_title
            )

        all_values = mean_shap_interactions_no_diag.flatten()
        nonzero_values = all_values[all_values > 0]

        vmin = np.min(nonzero_values)
        vmax = 1

        for value, i, j in top_pairs:
            ii, jj = i, j
            for _ in range(2):               
                if np.sign(self.spearmans[ii,jj]) != np.sign(self.pearsons[ii,jj]):
                    edge_color = 'black'
                else:
                    edge_color = "#3c8efa" if self.spearmans[ii][jj] < 0 else "#fa3c62"

                dashes = 1 if abs(self.spearmans[ii,jj]) < spearmans_threshold else 0

                pearson_val = self.pearsons[ii, jj]
                pearson_pval = self.pval_pearsons[ii, jj]

                spearman_val = self.spearmans[ii, jj]
                spearman_pval = self.pval_spearmans[ii, jj]

                edge_title = (
                    f"{self.feature_names[ii]} --> {self.feature_names[jj]}\n"
                    f"Pearson's r: {pearson_val:.2f} (pval {'< 0.0001' if pearson_pval < 0.0001 else '= ' + f'{pearson_pval:.4f}'})\n"
                    f"Spearman's r: {spearman_val:.2f} (pval {'< 0.0001' if spearman_pval < 0.0001 else '= ' + f'{spearman_pval:.4f}'})\n"
                    f"Mean absolute SHAP: {str(round(value,2))}"
                )

                def scale_width(value, vmin, vmax, min_width=0.5, max_width=6):
                    if vmax == vmin:
                        return (min_width + max_width) / 2
                    return min_width + (value - vmin) / (vmax - vmin) * (max_width - min_width)

                edge = {
                    'from': self.feature_names[ii],
                    'to': self.feature_names[jj],
                    'weight': round(float(mean_shap_interactions_norm[ii][jj]), 4),
                    'width': round(float(scale_width(mean_shap_interactions_norm[ii][jj], vmin, vmax)), 2),
                    'color': edge_color,
                    'title': edge_title,
                    'dashes': dashes
                }

                self.edges_data.append(edge)

                if mean_shap_interactions_norm[ii][jj] > threshold:
                    default_edges.append(edge)
                
                ii, jj = j, i

        for edge in default_edges:
            net.add_edge(
                edge['from'],
                edge['to'],
                color=edge['color'],
                width=edge['width'],
                dashes=True if edge['dashes']==1 else False,
                title=edge['title']
            )

        net.set_edge_smooth('dynamic')
        net.toggle_physics(True)
        net.show_buttons(filter_=["physics"])

        html_string = net.generate_html()

        modify_html_file(html_string, filename, self.edges_data)
