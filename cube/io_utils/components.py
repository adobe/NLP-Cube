from cube.data.objects import Doc, Sentence, Word, Token


class ComponentType:
    Tokenizer = 1
    CWExpander = 2
    POSTagger = 3
    Lemmatizer = 4
    Parser = 5
    NER = 6


class Model:
    def __init__(self):
        pass

    def __call__(self, input_object, **kwargs):
        return None


class Component():
    def __init__(self, use_gpu: bool = True, gpu_batch_size: int = 1):
        self.input_format = Doc  # type of data object or str
        self.output_format = Doc  # type of data object
        self.depends = []  # list of other components
        self.provides = []  # list of other components

        self.model_filepath = None

        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size

        self.model = None

    def load_model(self, model_path):
        pass

    def process(self, input_object):
        assert (self.model is not None), "Model is none, please load model first"
        return self.model(input_object=input_object)


class TokenizerComponent(Component):
    def __init__(self):
        super().__init__()
        self.input_format = str
        self.depends = []
        self.provides = [ComponentType.Tokenizer]


class CWExpanderComponent(Component):
    def __init__(self):
        super().__init__()
        self.depends = [ComponentType.Tokenizer]
        self.provides = [ComponentType.CWExpander]


class POSTaggerComponent(Component):
    def __init__(self):
        super().__init__()
        self.depends = [ComponentType.Tokenizer, ComponentType.CWExpander]
        self.provides = [ComponentType.POSTagger]


class LemmatizerComponent(Component):
    def __init__(self):
        super().__init__()
        self.depends = [ComponentType.Tokenizer, ComponentType.CWExpander, ComponentType.Parser]
        self.provides = [ComponentType.Lemmatizer]


class ParserComponent(Component):
    def __init__(self):
        super().__init__()
        self.depends = [ComponentType.Tokenizer, ComponentType.CWExpander]
        self.provides = [ComponentType.Parser]


class NERComponent(Component):
    def __init__(self):
        super().__init__()
        self.depends = [ComponentType.Tokenizer, ComponentType.CWExpander, ComponentType.Parser]
        self.provides = [ComponentType.NER]


class Pipeline():
    def is_valid(components: Component):
        available = set()
        required = set()
        for component in components:
            available |= set(component.provides)
            required |= set(component.depends)
        return required.issubset(available)
