from dyda.core import dyda_base


class DataSelectorBase(dyda_base.TrainerBase):
    """
    DataSelectorBase.input_data
        The first element of the list is numpy.arrays data to be selected.
    DataSelectorBase.output_data
        True to determine DataSelectorBase.metadata[0] is selected.
    DataSelectorBase.results
        {
        'base_name':
            Base_name of this results which may differ from
            DataSelectorBase.metadata[0] due to dalay.
        'is_key':
            True to determine DataSelectorBase.results['base_name']
            is selected.
        'ref_data_name':
            The name of reference data which would be assigned to
            the data not selected.
        }
    """

    def __init__(self, dtype="image", dyda_config_path=''):
        """ Init function of DataSelectorBase

        Keyword arguments:
        dtype -- data type
                 * array: non-image numpy array


        """
        super(DataSelectorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.results['base_name'] = ''
        self.results['is_key'] = False
        self.results['ref_data_name'] = ''

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def reset_results(self):
        """ Reset results """
        pass
