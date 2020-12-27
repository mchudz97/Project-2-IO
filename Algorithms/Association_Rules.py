from mlxtend.frequent_patterns import apriori, association_rules


class AssociationRules:
    def __init__(self, ohe_df):
        self.freq_items = apriori(ohe_df, min_support=.005, use_colnames=True, verbose=1)
        self.rules = association_rules(self.freq_items, metric='confidence', min_threshold=0.8)

