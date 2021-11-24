aws s3 cp py/company_simple_LDA.py s3://oaknorth-ml-dev-eu-west-1/company2vec/scripts/

aws emr add-steps --region eu-west-1 --cluster-id $1 --steps Type=spark,Name=ALBAJobNotLemmatized,Args=[--deploy-mode,cluster,--master,yarn,--conf,spark.yarn.submit.waitAppCompletion=true,--conf,spark.executor.memory=47696M,--conf,spark.driver.memory=47696M,--conf,spark.executor.cores=4,s3://oaknorth-ml-dev-eu-west-1/company2vec/scripts/company_simple_LDA.py,--word2id-path,s3://oaknorth-ml-dev-eu-west-1/company2vec/common/bow_description_not_lemmatized,--data-path,s3://oaknorth-ml-dev-eu-west-1/company2vec/data_desc_only/raw_company_features_bow_description_not_lemmatized,--model-path,s3://oaknorth-ml-dev-eu-west-1/company2vec/model/probabilistic_not_lemmatized/,--bow-column,bow_description_not_lemmatized],ActionOnFailure=CONTINUE