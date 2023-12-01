"""
PyDCI Notation

LaTex strings for plots to keep notation consistent
"""
from pydci.log import logger

DEF_NAMES = {'in': 'init',
            'up': 'update',
            'pr': 'pred',
            'ob': 'obs',
            'pf': 'pf-up',
            'mud': 'MUD',
            'true': r'\dagger'}
            

def gen_value_str(
    base: str = '\\lambda',
    sample: int = None,
    idx: int = None, 
    name: float = None,
    iteration: int = None,
    bold: bool = True,
    wrap: bool = True) -> str:
    """
    Generates a LaTeX-formatted parameter/observable string with optional superscripts or subscripts.

    General pyDCI value notation (for values such as parameters, observables, etc.) the following conventions:

    - The base symbol is a Greek letter, e.g. '\\lambda'.
    - The superscript is the sample number, e.g. '\\lambda^{(2)}'.
    - The subscript is the parameter index, e.g. '\\lambda_{1}'.
    - The name is a string, e.g. '\\lambda^{MUD}', can indicate special values
    - The iteration is an integer, e.g. '\\lambda^{MUD,3}', can indicate iterative updates/values

    Parameters
    ----------
    base : str, optional
        The base symbol for the parameter, by default '\\lambda'.
    sample: int, optional
        The superscript for the parameter, by default None. If provided, it's enclosed in parentheses.
    param: int, optional
        The subscript for the parameter, by default None.
    name: float, optional
        Name to be used either as a superscript if 'iteration' is provided, or
        as an assignment value, by default None.
    iteration : int, optional
        If provided along with 'name', it indicates the iteration and both are used
        as superscripts, separated by a comma, by default None.
    bold : bool, optional
        If True, the string is wrapped in a LaTeX bold command, by default True.
    wrap : bool, optional
        If True, the string is wrapped in LaTeX math mode, by default True.

    Returns
    -------
    str
        A string with the LaTeX-formatted parameter.

    Examples
    --------
    >>> gen_value_str()
    '$\lambda$'
    >>> param(base='x', sample=2)
    '$\lambda^{(2)}$'
    >>> gen_value_str(base='x', param_i=1)
    '$\lambda_{1}$'
    >>> gen_value_str(base='x', name='MUD')
    '$\lambda^{MUD}$'
    >>> gen_value_str(base='x', name='MUD', iteration=3)
    '$\lambda^{MUD,3}$'

    Notes
    -----
    The function uses raw f-strings (rf-string) to handle the curly braces which are part of LaTeX syntax.
    """
    param_str = rf'{base}' if not bold else rf'\mathbf{{{base}}}'
    if sample is not None:
        if iteration is not None:
            param_str += rf'^{{({samples}),{iteration}}}'
        else:
            param_str += rf'^{{({sample})}}'
    elif name is not None:
        if isinstance(name, str):
            name = rf'\mathrm{{{name}}}'
        if iteration is not None:
            param_str += rf'^{{{name},{iteration}}}'
        else:
            param_str += rf'^{{{name}}}'
    else:
        if iteration is not None:
            param_str += rf'^{{{iteration}}}'

    if idx is not None:
        param_str += rf'_{{{idx}}}'

    return rf'${param_str}$' if wrap else param_str


def gen_dist_str(
    base: str = '\pi',
    name: str = None,
    idx: str = None,
    iteration: str = None,
    arg: str = None,
    wrap: bool = True
) -> str:
    """
    Get LaTeX formatted string for distribution.

    Parameters
    ----------
    base : str, optional
        The base string for the distribution (default: '\\pi').
    name : str, optional
        The name of the distribution (default: None).
    idx : str, optional
        The index string (default: None).
    iteration : str, optional
        The iteration string (default: None).
    arg : str, optional
        The argument string (default: None).
    wrap : bool, optional
        Whether to wrap the LaTeX string in '$' (default: True).

    Returns
    -------
    str
        The LaTeX formatted string for the distribution.

    Examples
    --------
    >>> gen_dist_str()
    '$\\pi$'
    >>> gen_dist_str(name='X', idx='i', iteration=2, arg='x', wrap=False)
    '\\pi_{X}^{(2)}|_{i}(x)'
    >>> gen_dist_str(base='\\theta', idx='j', wrap=False)
    '\\theta|_{j}'
    """
    dist_str = rf'{base}'
    if name is not None:
        if isinstance(name, str):
            name = rf'\mathrm{{{name}}}'
        dist_str += rf'_{{{name}}}'
    if iteration is not None:    
        dist_str += rf'^{{({iteration})}}'
    if idx is not None:
        dist_str += rf''
        # TODO: how to indicate marginals?
        # logger.warning('Notation for marginal densities tbd.')
        # dist_str += rf'|_{{{idx}}}'
    if arg is not None:
        dist_str += rf'({arg})'
    
    return f"${dist_str}$" if wrap else dist_str


def gen_fun_str(
    base: str = 'Q',
    name: str = None,
    idx: str = None,
    iteration: str = None,
    arg: str = None,
    wrap: bool = True
) -> str:
    """
    Get LaTeX formatted string for function.

    Parameters
    ----------
    base : str, optional
        The base string for the function (default: 'Q').
    name : str, optional
        The name of the function (default: None).
    idx : str, optional
        The index string (default: None).
    iteration : str, optional
        The iteration string (default: None).
    arg : str, optional
        The argument string (default: None).
    wrap : bool, optional
        Whether to wrap the LaTeX string in '$' (default: True).

    Returns
    -------
    str
        The LaTeX formatted string for the function.

    Examples
    --------
    >>> gen_fun_str()
    '$Q$'
    >>> gen_fun_str(name='F', idx='i', iteration=2, arg='x', wrap=False)
    'Q_{F}^{(2)}(x)'
    >>> gen_fun_str(base='f', idx='j', wrap=False)
    'f_{j}'
    """
    fun_str = rf'{base}'
    if name is not None:
        if isinstance(name, str):
            name = rf'\mathrm{{{name}}}'
        fun_str += rf'_{{{name}}}'
    if iteration is not None:
        fun_str += rf'^{{({iteration})}}'
    if idx is not None:
        fun_str = rf'({fun_str})_{{{idx}}}'
    if arg is not None:
        fun_str += rf'({arg})'
    
    return f"${fun_str}$" if wrap else fun_str

def gen_scalar_str(
    base: str,
    iteration: str = None,
    sample: int = None,
    wrap: bool = True,
):
    """
    Primarily for update ratios and weights on samples

    No idx, since a scalar value, and no real special values.
    Just sample (superscript, paranthesis) and iteration (superscript, after a comma)
    modifications for scalar values now.
    """
    scalar_str = rf'{base}'
    if sample is not None:
        if iteration is not None:
            scalar_str += rf'^{{({sample}), {iteration}}}'
        else:
            scalar_str += rf'^{{({sample})}}'
    
    return f"${scalar_str}$" if wrap else scalar_str


def gen_latex(
    key: str,
    idx: str = None,
    iteration: str = None,
    sample: int = None,
    name: float = None,
    bold: bool = True,
    wrap: bool = True,
    arg: str = None,
) -> str:
    """
    Generate LaTeX formatted string for various components based on the specified key.

    Parameters
    ----------
    key : str
        The key specifying the type of component ('param', 'q', 'dist', or 'Q').
    idx : str, optional
        The index string (default: None).
    iteration : str, optional
        The iteration string (default: None).
    sample : int, optional
        The sample number (default: None).
    name : float, optional
        The name of the (default: None).
    bold : bool, optional
        Whether to make the generated string bold (default: True).
    wrap : bool, optional
        Whether to wrap the LaTeX string in '$' (default: True).
    arg : str, optional
        The argument string (default: None).

    Returns
    -------
    str
        The LaTeX formatted string for the specified component.

    Examples
    --------
    >>> gen_latex('param', idx='i', iteration=2, sample=1, name=0.5, bold=False, wrap=False)
    '\\lambda_{i}^{(2)}^{(1)}_{0.5}'
    >>> gen_latex('q', idx='j', sample=3, name=1.0, wrap=False)
    'q_{j}^{(3)}_{1.0}'
    >>> gen_latex('dist', name='X', idx='k', arg='x', wrap=False)
    '\\pi_{X}|_{k}(x)'
    >>> gen_latex('Q', name='F', idx='l', iteration=4, arg='y', wrap=False)
    'Q_{F}^{(4)}(y)'
    """
    # One index everything
    if isinstance(iteration, int):
        iteration += 1
    if isinstance(idx, int):
        idx += 1
    if isinstance(sample, int):
        sample += 1

    if isinstance(arg, dict):
        sub_key = arg.pop('key')
        arg['wrap'] = False
        arg = gen_latex(sub_key, **arg)
    elif isinstance(arg, str):
        arg = unwrap(arg)

    if key == 'param':
        return gen_value_str(
            base='\\lambda',
            idx=idx,
            iteration=iteration,
            sample=sample,
            name=name,
            bold=bold,
            wrap=wrap,
        )
    elif key == 'q':
        return gen_value_str(
            base='q',
            idx=idx,
            iteration=iteration,
            sample=sample,
            name=name,
            bold=bold,
            wrap=wrap,
        )
    elif key == 'dist':
        return gen_dist_str(
            base='\pi',
            name=name,
            idx=idx,
            iteration=iteration,
            arg=arg,
            wrap=wrap,
        )
    elif key == 'Q':
        return gen_fun_str(
            base='Q',
            name=name,
            idx=idx,
            iteration=iteration,
            arg=arg,
            wrap=wrap,
        )
    elif key == 'r':
        return gen_scalar_str(
            base='r',
            sample=sample,
            iteration=iteration,
            wrap=wrap,
        )
    elif key == 'w':
        return gen_scalar_str(
            base='w',
            sample=sample,
            iteration=iteration,
            wrap=wrap,
        )
    else:
        # Default to a value (ector or scalar)
        return gen_value_str(
            base=key,
            name=name,
            idx=idx,
            iteration=iteration,
            wrap=wrap,
            bold=bold,
        )


def unwrap(s: str) -> str:
    """
    Removes the $ character from the start and end of a string if it is wrapped in $.

    Args:
        s: The string to process.

    Returns:
        The processed string.
    """
    if s.startswith('$') and s.endswith('$'):
        return s[1:-1]
    else:
        return s


def mud_pt(
    iteration=None,
    idx=None,
):
    return gen_latex(
        'param',
        iteration=iteration,
        idx=idx,
        name='MUD',
        bold=True,
    )

def Q(
    iteration=None,
    idx=None,
    name=None,
    bold=True,
    arg=None,
):
    return gen_latex(
        'Q',
        iteration=iteration,
        idx=idx,
        name=name,
        arg=arg,
    )


def q_pca(
    iteration=None,
    idx=None,
    bold=True,
    arg=None,
):
    return gen_latex(
        'Q',
        iteration=iteration,
        idx=idx,
        name='PCA',
        arg=arg,
    )

def q(
    iteration=None,
    idx=None,
    name=None,
    bold=True,
):
    return gen_latex(
        'q',
        iteration=iteration,
        idx=idx,
        name=name,
        bold=bold,
    )

def lam(
    iteration=None,
    idx=None,
    name=None,
    bold=True,
):
    return gen_latex(
        'param',
        iteration=iteration,
        idx=idx,
        name=name,
        bold=bold,
    )

def r(
    iteration=None,
    sample=None,
    weighted=False,
    wrap=True,
):
    """
    Update ratio string

    Examples
    --------
    >>> r()
    '$r$'
    >>> r(weighted=True)
    '$wr$'
    >>> r(iteration=1, sample=2, weighted=False)
    r'$r^{{(2),1}}$'
    >>> r(sample=2, weighted=False)
    r'$r^{{(2)}}$'
    >>> r(iteration=1, weighted=False)
    r'$r^{{(i),1}}$'
    >>> r(iteration=1, sample=2, weighted=True)
    r'$w^{{(2),1}}r^{{(2),1}}$'
    >>> r(sample=2, weighted=True)
    r'$w^{{(2)}}r^{{(2)}}$'
    >>> r(iteration=1, weighted=True)
    r'$w^{{(i),1}}r^{{(i),1}}$'
    """
    r_str = 'r'
    if weighted:
        if iteration:
            sample = 'i' if sample is None else sample
            r_str = f"w^{{({sample}),{iteration}}}r^{{({sample}),{iteration}}}"
        elif iteration is None and sample:
            r_str = f"w^{{({sample})}}r^{{({sample})}}"
        else:
            r_str = 'wr'
    else:
        if iteration:
            sample = 'i' if sample is None else sample
            r_str = f"r^{{({sample}),{iteration}}}"
        elif iteration is None and sample:
            r_str = f"r^{{({sample})}}"
    
    return f"${r_str}$" if wrap else r_str  

# def r(
#     iteration=None,
#     sample=None,
#     weighted=False,
#     wrap=True,
# ):
#     """
#     Update ratio string
# 
#     Examples
#     --------
#     >>> r()
#     '$r$'
#     >>> r(weighted=True)
#     '$wr$'
#     >>> r(iteration=1, sample=2, weighted=False)
#     r'$r^{{(2),1}}$'
#     >>> r(sample=2, weighted=False)
#     r'$r^{{(2)}}$'
#     >>> r(iteration=1, weighted=False)
#     r'$r^{{(i),1}}$'
#     >>> r(iteration=1, sample=2, weighted=True)
#     r'$w^{{(2),1}}r^{{(2),1}}$'
#     >>> r(sample=2, weighted=True)
#     r'$w^{{(2)}}r^{{(2)}}$'
#     >>> r(iteration=1, weighted=True)
#     r'$w^{{(i),1}}r^{{(i),1}}$'
#     """
#     pass

def pi(
    name,
    iteration=None,
    idx=None,
    arg=None,
):
    """
    Entry point for generating distribution strings
    """
    if name not in DEF_NAMES.keys():
        raise ValueError(f'Unknown distribution name: {name}')

    return gen_latex(
        'dist',
        iteration=iteration,
        idx=idx,
        name=DEF_NAMES[name],
        arg=arg,
    )

def e_r(value=None, format_spec=".3f", iteration=None, sample=None, wrap=True):
    """
    Expected ratio string
    """
    r_str = r(iteration=iteration, sample=sample, wrap=False)
    if iteration is not None:
        r_str = rf"{w(iteration=iteration, sample=sample, wrap=False)}{r_str}" if iteration is not None else ""

    e_r_str = rf"\mathbb{{E}}({r_str})"

    if value is not None:
        e_r_str = rf"{e_r_str} = {value:{format_spec}}"
    
    return f"${e_r_str}$" if wrap else e_r_str  


def exp_ratio_str(e_r, format_spec=".3f", iteration=None):
    """
    Expected ratio string
    """
    iteration_str = rf"^{{({iteration})}}" if iteration is not None else ""
    return rf"$\mathbb{{E}}(r{iteration_str})$= {e_r:{format_spec}}"

def kl_str(arg_type=None, wrap=True, value=None, format_spec=".3f"):
    """
    KL string
    """
    kl_str = rf"\mathrm{{KL}}_{{\mathrm{{DCI}}}}"
    if arg_type is not None:
        if arg_type == 'up':
            kl_str += rf"({pi_up(iteration=iteration, wrap=False)} : {pi_init(iteration=iteration, wrap=False)})" 
        if arg_type == 'obs':
            kl_str += rf"({pi_obs(iteration=iteration, wrap=False)} : {pi_pr(iteration=iteration, wrap=False)})" 

    if value is not None:
        kl_str = rf"{kl_str } = {value:{format_spec}}"

    return f"${kl_str}$" if wrap else kl_str


def pi_up(
        idx: str = None,
        iteration: str = None,
        sample: int = None,
        name: float = None,
        bold: bool = True,
        wrap: bool = True,
        arg: str = None,
):
    return gen_latex(
        'dist',
        idx=idx,
        iteration=iteration,
        sample=sample,
        name='update',
        bold=False,
        wrap=wrap,
        arg=arg,
    )