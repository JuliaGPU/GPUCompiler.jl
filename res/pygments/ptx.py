from pygments.lexer import RegexLexer, include, words
from pygments.token import *

# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

class CustomLexer(RegexLexer):
    string = r'"[^"]*?"'

    followsym = r'[a-zA-Z0-9_$]*'
    identifier = r'(?:[a-zA-Z]' + followsym + r'| [_$%]' + followsym + r')'

    tokens = {
        'root': [
            include('whitespace'),
            (r'%' + identifier, Name.Variable),

            include('definition'),
            include('statement'),
            include('type'),

            (identifier, Name.Variable),

            (r'(\d+\.\d*|\.\d+|\d+)[eE][+-]?\d+[LlUu]*', Number.Float),
            (r'(\d+\.\d*|\.\d+|\d+[fF])[fF]?', Number.Float),
            (r'0x[0-9a-fA-F]+[LlUu]*', Number.Hex),
            (r'0[0-7]+[LlUu]*', Number.Oct),
            (r'\b\d+[LlUu]*\b', Number.Integer),

            (r'[&|^+*/%=~-]', Operator),
            (r'[()\[\]\{\},.;<>@]', Punctuation),
        ],
        'whitespace': [
            (r'(\n|\s)+', Text),
            (r'/\*.*?\*/', Comment.Multiline),
            (r'//.*?\n', Comment.Single),
        ],
        'definition': [
            (words(('func', 'reg'), prefix=r'\.', suffix=r'\b'), Keyword.Reserved),
            (r'^' + identifier + r':', Name.Label),
        ],
        'statement': [
            # directive
            (words((
                'address_size', 'file', 'minnctapersm', 'target', 'align', 'func', 'param',
                'tex', 'branchtarget', 'global', 'pragma', 'version', 'callprototype',
                'loc', 'reg', 'visible', 'calltargets', 'local', 'reqntid', 'weak', 'const',
                'maxnctapersm', 'section', 'entry', 'maxnreg', 'shared', 'extern',
                'maxntid', 'sreg', ), prefix=r'\.', suffix=r'\b'), Keyword),
            # instruction
            (words((
                'abs', 'div', 'or', 'sin', 'add', 'ex2', 'pmevent', 'slct', 'vmad', 'addc',
                'exit', 'popc', 'sqrt', 'vmax', 'and', 'fma', 'prefetch', 'st', 'atom',
                'isspacep', 'prefetchu', 'sub', 'vmin', 'bar', 'ld', 'prmt', 'subc', 'bfe',
                'ldu', 'rcp', 'suld', 'vote', 'bfi', 'lg2', 'red', 'suq', 'vset', 'bfind',
                'mad', 'rem', 'sured', 'bret', 'sust', 'vshl', 'brev', 'madc', 'rsqrt',
                'testp', 'vshr', 'brkpt', 'max', 'sad', 'tex', 'vsub', 'call', 'membar',
                'selp', 'tld4', 'clz', 'min', 'set', 'trap', 'xor', 'cnot', 'mov', 'setp',
                'txq', 'copysign', 'mul', 'shf', 'vabsdiff', 'cos', 'shfl', 'cvta', 'not',
                'shr', 'cvt', 'neg', 'shl', 'vadd'), prefix=r'\b', suffix=r'[\.\w]+\b'), Keyword),
            (words((
                'vavrg', 'vmax', 'vmin', 'vset', 'mad', 'vsub', 'mul', 'vabsdiff',
                'vadd'), prefix=r'\b', suffix=r'[24]\b'), Keyword),
        ],
        'type': [
            (words((
                's8', 's16', 's32', 's64',
                'u8', 'u16', 'u32', 'u64',
                'f16', 'f16x2', 'f32', 'f64',
                'b8', 'b16', 'b32', 'b64',
                'pred'), prefix=r'\.', suffix=r'\b'), Keyword.Type),
        ],
    }
