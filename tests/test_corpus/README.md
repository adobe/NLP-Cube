**Note: This is a subset of the full UD_English-EWT corpus, fit for testing purposes. Below is the original readme:**


Universal Dependencies - English Dependency Treebank
Universal Dependencies English Web Treebank v2.2 -- 2018-04-15
https://github.com/UniversalDependencies/UD_English-EWT

# Summary

A Gold Standard Universal Dependencies Corpus for English,
built over the source material of the English Web Treebank
LDC2012T13 (https://catalog.ldc.upenn.edu/LDC2012T13).


# Introduction

The corpus comprises 254,830 words and 16,622 sentences, taken from five genres
of web media: weblogs, newsgroups, emails, reviews, and Yahoo! answers. See the
LDC2012T13 documentation for more details on the sources of the sentences. The
trees were automatically converted into Stanford Dependencies and then
hand-corrected to Universal Dependencies. All the basic dependency annotations have been single-annotated, a limited portion of them have been double-annotated,
and subsequent correction has been done to improve consistency. Other aspects
of the treebank, such as Universal POS, features and enhanced dependencies, has mainly been done
automatically, with very limited hand-correction.

# License/Copyright

Universal Dependencies English Web Treebank © 2013-2018
by The Board of Trustees of The Leland Stanford Junior University.
All Rights Reserved.

The annotations and database rights of the Universal Dependencies
English Web Treebank are licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

The underlying texts come from various sources collected for the
LDC English Web Treebank. Some parts are in the public domain.
Portions may be © 2012 Google Inc., © 2011 Yahoo! Inc.,
© 2012 Trustees of the University of Pennsylvania and/or
© other original authors.


# Structure

This directory contains a corpus of sentences annotated using Universal
Dependencies annotation. The corpus comprises 254,830 words and 16,622
sentences, taken from various web media including weblogs, newsgroups, emails,
reviews, and Yahoo! answers; see the LDC2012T13 documentation for more details
on the source of the sentences.  The trees were automatically converted into
Stanford Dependencies and then hand-corrected to Universal Dependencies. All
the dependency annotations have been single-annotated, and a limited portion of
them have been double-annotated with interannotator agreement at approximately
96%. The sentence IDs include the genre and the filename
of the original LDC2012T13 filename.

This corpus is compatible with the CoNLL-U format defined for Universal
Dependencies. See:

   http://universaldependencies.github.io/docs/format.html

The dependency taxonomy can be found on the Universal Dependencies web site:

   http://www.universaldependencies.org

For the conversion to v2, we performed an automatic conversion with extensive
spot-checking, and manual adjudication of ambiguous cases.

The enhanced dependencies were automatically obtained by running an adapted version
of the converter by Schuster and Manning (2016). These dependencies have **not** been
manually checked.

# Deviations from UD

Version 2.2 of the English UD treebank conforms to the UD guidelines in
almost all respects, but there remain the following deviations:

 - The UD dependency `flat` is largely only used for person names.
 - Dates are not annotated consistently.
 - The attachment of punctuation tokens is sometimes not according to the
    guidelines.

# Changelog

**2018-04-15 v2.2**
  - Repository renamed from UD_English to UD_English-EWT
  - Automatically added enhanced dependencies (These have not been manually checked!)
  - Fixed some wrong lemmata and POS tags
  - Fixed miscellaneous syntactic issues

**2017-11-15 v2.1**

 - Fixed some wrong lemmata, POS tags
 - Fixed miscellaneous syntactic issues
 - Added basic dependencies into the `DEPS` column according to the CONLL-U v2
    format

**2017-02-15 v2.0**

 - Updated treebank to conform to v2 guidelines
 - Fixed some wrong lemmata
 - Fixed miscellaneous syntactic issues
 - Added empty nodes for gapped constructions in enhanced representation

**2016-11-15 v1.4**

 - Changed POS tag of fused det-noun pronouns (e.g., *"somebody"*, *"nothing"*)
    to `PRON`
 - Added original, untokenized sentences to CoNLL-U files
 - Fixed some POS errors, features and wrong lemmata
 - Fixed miscellaneous syntactic issues in a few sentences

**2016-05-15 v1.3**

 - Improved mapping of `WDT` to UPOS
 - Corrected lemma of *"n't"* to *"not"*
 - Fixed some errors between `advcl`, `ccomp` and `parataxis`
 - Fixed inconsistent analyses of sentences repeated between dev and train sets
 - Fixed miscellaneous syntactic issues in a few sentences


**2015-11-15 v1.2**

 - Bugfix: removed *_NFP* suffix from some lemmas
 - Fixed date annotations to adopt UD standard
 - Remove escaping of *(* and *)* from word tokens (XPOSTAGs are still `-LRB-`
    and `-RRB-`)
 - Improved precision of `xcomp` relation
 - Improved recall of `name` relation
 - Corrected lemmas for reduced auxiliaries
 - Corrected UPOS tags of pronominal uses of *this/that/these/those* (from `DET`
    to `PRON`)
 - Corrected UPOS tags of subordinating conjunctions (from `ADP` to `SCONJ`)
 - Corrected UPOS tags of some main verbs (from `AUX` to `VERB`)

# Contributing

To help improve the corpus, please alert us to any errors you find in it.
The best way to do this is to file a github issue at:

   https://github.com/UniversalDependencies/UD_English-EWT/issues

We also welcome pull requests. If you want to make edits, please modify the
trees in the idividual files in the `not-to-release/sources` directory instead
of making direct changes to `en-ud-{dev,test,train}.conllu`.

# Acknowledgments

Annotation of the Universal Dependencies English Web Treebank was carried out by
(in order of size of contribution):

 - Natalia Silveira
 - Timothy Dozat
 - Sebastian Schuster
 - Miriam Connor
 - Marie-Catherine de Marneffe
 - Nathan Schneider
 - Samuel Bowman
 - Hanzhi Zhu
 - Daniel Galbraith
 - Christopher Manning
 - John Bauer

Creation of the CoNLL-U files, including calculating UPOS, feature, and lemma
information was primarily done by

 - Sebastian Schuster
 - Natalia Silveira

The construction of the Universal Dependencies English Web Treebank was
partially funded by a gift from Google, Inc., which we gratefully acknowledge.


# Citations

You are encouraged to cite this paper if you use the Universal Dependencies
English Web Treebank:


    @inproceedings{silveira14gold,
      year = {2014},
      author = {Natalia Silveira and Timothy Dozat and Marie-Catherine de
		  Marneffe and Samuel Bowman and Miriam Connor and John Bauer and
		  Christopher D. Manning},
      title = {A Gold Standard Dependency Corpus for {E}nglish},
      booktitle = {Proceedings of the Ninth International Conference on Language
        Resources and Evaluation (LREC-2014)}
    }

# Metadata

```
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v1.0
License: CC BY-SA 4.0
Includes text: yes
Genre: blog social reviews email
Lemmas: automatic with corrections
UPOS: converted with corrections
XPOS: manual native
Features: automatic
Relations: manual native
Contributors: Silveira, Natalia; Dozat, Timothy; Manning, Christopher; Schuster, Sebastian; Bauer, John; Connor, Miriam; de Marneffe, Marie-Catherine; Schneider, Nathan; Bowman, Sam; Zhu, Hanzhi; Galbraith, Daniel
Contributing: here source
Contact: syntacticdependencies@lists.stanford.edu
===============================================================================
```
