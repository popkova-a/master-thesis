import yaml


class ConfigParser:
    """
    Parser for configuration files containing model, data, and training information.

    Args:
        config_file_path (str): Path to the .yaml configuration file to be parsed.
    """

    def __init__(self,
                 config_file_path: str):
        self.config_file_path = config_file_path

    def parse_config_file(self) -> dict:
        """
        Parses the configuration file and extracts relevant sections.

        Returns:
            config_args (dict): Arguments from the config file.

         Raises:
            FileNotFoundError: If the configuration file cannot be found.
            yaml.YAMLError: If there is an error parsing the YAML file.
            KeyError: If required sections are missing from the configuration file.
        """

        # Parse the yaml file
        with open(self.config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Extract relevant sections
        config_args = dict()
        config_args['event_vision_model'] = config.get('event_vision_model')
        config_args['vision_language_model'] = config.get('vision_language_model')
        config_args['data'] = config.get('data')
        config_args['train'] = config.get('train')

        return config_args


def build_config_parser(config_file_path: str) -> ConfigParser:
    """
    Create a ConfigParser instance initialized with the given configuration file path.

    Args:
        config_file_path (str): Path to the .yaml configuration file to be parsed.

    Returns:
        ConfigParser: A ConfigParser instance loaded with the configuration from the specified file.
    """

    return ConfigParser(config_file_path=config_file_path)
