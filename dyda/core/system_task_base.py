from dyda.core import dyda_base


class SystemTaskBase(dyda_base.TrainerBase):

    def __init__(self, dyda_config_path=''):
        """ __init__ of SystemTaskBase """

        super(SystemTaskBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def reset_results(self):
        """ Reset results """
        pass
