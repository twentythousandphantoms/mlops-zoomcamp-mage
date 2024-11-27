if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    from sklearn.feature_extraction import DictVectorizer

    target_columns = ['PULocationID', 'DOLocationID']
    data_dicts = df[target_columns].to_dict(orient='records')  # Преобразование в список словарей

    vectorizer = DictVectorizer()  # Создаём объект DictVectorizer
    feature_matrix = vectorizer.fit_transform(data_dicts)  # Преобразуем данные в матрицу

    # Define the target variable (the column in the DataFrame containing the labels)
    target_column = 'duration'
    labels_train = df[target_column].values  # Extract target values as a NumPy array

    # Initialize the linear regression model
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()

    # Train the model using the training data (feature matrix and target labels)
    linear_model.fit(feature_matrix, labels_train)

    # Make predictions on the training data
    # predictions_train = linear_model.predict(feature_matrix)

    print(linear_model.intercept_)

    return vectorizer, linear_model


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'