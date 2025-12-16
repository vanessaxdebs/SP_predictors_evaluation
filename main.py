from scripts import (
    data_collection,
    data_preparation,
    data_analysis,
    pswm_training_set_preparation,
    pswm,
    svm,
    results_analysis,
)

if __name__ == '__main__':
    data_collection.main()
    data_preparation.main()
    data_analysis.main()
    pswm_training_set_preparation.main()
    pswm.main()
    svm.main()
    results_analysis.main()