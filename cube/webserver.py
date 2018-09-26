#
# Author: Tiberiu Boros
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys

sys.path.append("..")
import optparse
from flask import Flask
from flask import Response
from cube.api import Cube

app = Flask(__name__)
singletonServer = False

lang2cube = {}


@app.route('/')
def index():
    return "NLP-Cube server running. Use /help to learn more."


@app.route('/nlp', methods=['GET', 'POST'])
def nlp():
    from flask import request
    if request.args.get('text'):
        query = request.args.get('text')
    else:
        query = request.form.get('text')

    if request.args.get('lang'):
        lang = request.args.get('lang')
    else:
        lang = request.form.get('lang')

    if request.args.get('format'):
        format = request.args.get('format')
    else:
        format = request.form.get('format')

    if not format:
        format = 'CONLL'

    if format != 'CONLL' and format != 'JSON':
        return Response("Allowed values for 'format' are CONLL or JSON", mimetype='text/plain',
                        status=500)
    if not query or not lang:
        return Response("You need to specify the language (lang) and text (text) parameters", mimetype='text/plain',
                        status=500)

    if lang not in lang2cube:
        return Response("This language has not beed preloaded during server startup", mimetype='text/plain',
                        status=500)

    thecube = lang2cube[lang]
    result = thecube(query)
    if format == 'CONLL':
        text = ""
        for seq in result:
            for entry in seq:
                text += str(
                    entry.index) + "\t" + entry.word + "\t" + entry.lemma + "\t" + entry.upos + "\t" + entry.xpos + "\t" + entry.attrs + "\t" + str(
                    entry.head) + "\t" + entry.label + "\t" + entry.deps + "\t" + entry.space_after + "\n"
            text += "\n"
        return Response(text, mimetype='text/plain',
                        status=200)
    else:
        import json
        new_seqs = []
        for seq in result:
            new_seq = []
            for entry in seq:
                new_seq.append(entry.__dict__)
            new_seqs.append(new_seq)
        text = json.dumps(new_seqs, sort_keys=False, indent=4)

        return Response(text, mimetype='application/json',
                        status=200)


@app.route('/help', methods=['GET', 'POST'])
def help():
    text = "NLP-Cube server\n\n" \
           "Use /nlp endpoint to process any text.\n" \
           "\tParameters:\n" \
           "\t\t lang - language code\n" \
           "\t\t text - text to process\n" \
           "\t\t format - output format for data: CONLL|JSON (default is CONLL with plain/text output)"
    return Response(text, mimetype='text/plain',
                    status=200)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--port', action='store', dest='port', type='int', default=8080,
                      help='Binding port for web service (default: 8080)')
    parser.add_option('--host', action='store', dest='host', default='0.0.0.0',
                      help='Binding IP for server (default: 0.0.0.0)')
    parser.add_option('--lang', action='append', dest='languages', default=[],
                      help='Preload language. You can use this param multiple times: --lang en --lang fr ... (default is just ["en"])')

    (params, _) = parser.parse_args(sys.argv)

    if len(params.languages) == 0:
        params.languages = ['en']

    for lang in params.languages:
        lang2cube[lang] = Cube(verbose=True)
        lang2cube[lang].load(lang)

    app.run(port=params.port, host=params.host)
