import utils
import pandas as pd
import random

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
#import seaborn as sns

import os
import uuid

from google.cloud import bigquery
from google.oauth2 import service_account


import cv2
import keras
from keras import backend as K
from keras.models import Model
from keras.models import load_model

credentials = service_account.Credentials.from_service_account_file(
    'credentials json file ')
project_id = 'project_id'
client = bigquery.Client(credentials= credentials,project=project_id)


def plot_mask_on_img(old_img):

    tdf = pd.read_csv('mask_test.csv', index_col=0)
    tdf = tdf.fillna('')

    plt.figure(figsize=[25, 20])
    types = ''
    for index, row in tdf.iterrows():
        img = cv2.imread(old_img)
        img = cv2.resize(img, (525, 350))
        mask_rle = row['EncodedPixels']
        plt.subplot(2, 2, index + 1)
        plt.imshow(img)
        if mask_rle != '':
            types = types + ', ' + row['Image_Label'].split('_')[-1]

        plt.imshow(utils.rle2mask(mask_rle, img.shape), alpha=0.5, cmap='gray')
        plt.title(row['Image_Label'].split('_')[-1], fontsize=25)
        plt.axis('off')

    types_all = types[2:]
    after_mask = ''.join(str(uuid.uuid4()).split('-')) + '.png'
    plt.savefig('static/upload/' + after_mask)
    plt.close()

    imgId = str(random.randint(22200,24200))+'.jpg'
    print(imgId)
    for i in types_all.split(', '):
        query = """INSERT INTO `project_id.project_satellite.report`
                   (int64_field_0 , ImageId, Label, exist, Frequent, types )
                   VALUES (2, @d, @a, 1, @b, @c)"""

        query_params = [
            bigquery.ScalarQueryParameter("d", "STRING", imgId),
            bigquery.ScalarQueryParameter("a", "STRING", i),
            bigquery.ScalarQueryParameter("b", "STRING", types_all),
            bigquery.ScalarQueryParameter("c", "INT64", len(types_all.split(', '))),
        ]

        job_config = bigquery.QueryJobConfig()
        job_config.query_parameters = query_params

        client.query(
            query,
            # Location must match that of the dataset(s) referenced in the query.
            location="US",
            job_config=job_config,
        )

    return after_mask





def segmentation(name,path,model):
    test_df = []
    sub_df = pd.DataFrame(data={'Image_Label': [name+"_Fish", name+"_Flower",
                                                name+"_Gravel", name+"_Sugar"],
                                'EncodedPixels': ['1 1','1 1','1 1','1 1'],'ImageId':[name, name,
                                                                                        name, name]})

    test_imgs=pd.DataFrame(data={'ImageId': [name]})
    test_generator = utils.DataGenerator(
        [0],
        df=test_imgs,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=(320, 480),
        n_channels=3,
        base_path=path,
        target_df=sub_df,
        batch_size=1,
        n_classes=4
    )

    batch_pred_masks = model.predict_generator(
        test_generator,
        workers=1,
        verbose=1
    )

    for j, b in enumerate([0]):
        filename = test_imgs['ImageId'].iloc[b]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        pred_masks = batch_pred_masks[j, ].round().astype(int)
        pred_rles = utils.build_rles(pred_masks, reshape=(350, 525))

        image_df['EncodedPixels'] = pred_rles
        test_df.append(image_df)

    test_df[0].iloc[:,:2].to_csv('./mask_test.csv')

    return test_df





def process(old_img):
    print('start load model')
    model = load_model('./models/unet.h5', custom_objects={'bce_dice_loss': utils.bce_dice_loss, 
                                                           'dice_coef': utils.dice_coef})
    print('finish load model')

    segmentation(old_img.split('/')[-1], './static/upload',model)
    K.clear_session()


    return plot_mask_on_img(old_img)