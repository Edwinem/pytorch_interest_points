import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def json_remove_comments(json_file):
    '''
    Remove comments from a json file
    :param json_file:
    :return:
    '''
    try:
        from json_minify import json_minify
        json_out=''
        for line in json_file:  # Read it all in
            json_out += line
        return json_minify(json_out)
    except ImportError:
        json_out=''
        for line in json_file:  # Read it all in
            json_out += line
        almost_json=remove_comments(json_out)
        proper_json=remove_trailing_commas(almost_json)
        return proper_json



import re
try:
    import ujson as json # Speedup if present; no big deal if not
except ImportError:
    import json

def remove_comments(json_like):
    """
    Removes C-style comments from *json_like* and returns the result.  Example::
        >>> test_json = '''\
        {
            "foo": "bar", // This is a single-line comment
            "baz": "blah" /* Multi-line
            Comment */
        }'''
        >>> remove_comments('{"foo":"bar","baz":"blah",}')
        '{\n    "foo":"bar",\n    "baz":"blah"\n}'
    """
    comments_re = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    def replacer(match):
        s = match.group(0)
        if s[0] == '/': return ""
        return s
    return comments_re.sub(replacer, json_like)

def remove_trailing_commas(json_like):
    """
    Removes trailing commas from *json_like* and returns the result.  Example::
        >>> remove_trailing_commas('{"foo":"bar","baz":["blah",],}')
        '{"foo":"bar","baz":["blah"]}'
    """
    trailing_object_commas_re = re.compile(
        r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    trailing_array_commas_re = re.compile(
        r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    # Fix objects {} first

    #this can fail on UNICODE. Dont know if I care
    objects_fixed=trailing_object_commas_re.sub("}", json_like)
    # Now fix arrays/lists [] and return the result
    return trailing_array_commas_re.sub("]", objects_fixed)
