import pandas as pd
import timm


def display_model_list():
    model_names = ['*resnet*', '*efficientnet*', '*convnext*']

    for _, name in enumerate(model_names):
        models = timm.list_models(name, pretrained=True)
        df = pd.DataFrame(models, columns=['Model Name'])
        print(df.head(20))
