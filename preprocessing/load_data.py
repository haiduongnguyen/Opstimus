class DataLoader():

    def __init__(self, data, label_col=None) -> None:
        self.data = data 
        self.label_col = label_col


    def get_features(self):
        if self.label_col  and self.label_col != "": 
            if self.label_col in self.data.columns.to_list():
                X = self.data.drop(self.label_col, axis=1)
                return X
            else:
                raise ValueError("Label column mus exists in data")
        else:
            return self.data 
        
    def get_labels(self):
        """
        this function return labels, use in evaluation only
        """
        if self.label_col and self.label_col != "":
            if self.label_col in self.data.columns.to_list():
                return self.data[self.label_col]
            else:
                raise ValueError("Label column mus exists in data")
        else:
            raise ValueError("This data has no label column")
        
    def remove_columns(self, cols):
        self.data = self.data.drop(columns=cols)


    def train_test_split(self, test_size=0.2, random_state=None):
        from sklearn.model_selection import train_test_split

        X = self.get_features()
        if self.label_col and self.label_col != "":
            y = self.get_labels()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, None, None