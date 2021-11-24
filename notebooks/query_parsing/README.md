# Query parsing

The purpose of this notebook is to prototype query parsing functionality. The idea is to be able to parse a
free text query into a query plan that elastic search can understand and use to obtain more focused results
E.g. `steel companies in China with more than 5 employees` will restrict to only companies in China, having steel in the industry
type description and with more than 5 employees.

### CFGs and free text parsing

In order to obtain an elegant mapping from real words to textual fields, we begin by specifying a very loose grammar that
can be used to 'generate' queries. We parse free text queries using this grammar and use the resulting syntax tree to populate
respective fields in the call to elastic search. Because free text is, well, free, this grammar would need to be hugely complex
in order to work in practice. To keep it manageable, we apply a two-step process. Firstly we preprocess queries using syntax taggers
from [flair](https://github.com/zalandoresearch/flair) and then we use this simplified free text to construct a syntax tree using a
manageable grammar.

#### Query preprocessing

For this step we rely on two sequence taggers provided by `flair`. Firstly, we use the basic `POS` tagger to indentify all cardinals
in the input sentence and convert them to a placeholder `CARDINAL` symbol. Using this new query, we search for comparator operators 
followed by a cardinal, and replace them with their mathematical equivalent (i.e. >, >=, <, <=). For the time being we use a
hand-created list of symbols to identify these comparisons (e.g. 'more than' -> >), since there seems to be no good tagger to do this
automatically.

Next, we use this sanitized sentence together with the [ner-ontonomes](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf) tagger to identify  locations (regions) and countries. We replace them with the `LOC` and `GPE` 
placeholders respectively.

#### CFG and getting a syntax tree

From the above we have a relatively tame sentence that we can meaningfully process using a small (to code) grammar. Care needs to be 
taken when handling industry type description, as we want the query to be useful without the analysts knowing the exact (and 
possibly very verbose and hard to remember) industry description. To do this, we retrieve all descriptions from the elastic search
index, and split them into a set of keywords, which we use in the grammar as described below.

On a very high level, the grammar comprises a start symbol `CP` (for Company Production rule). The first transform splits this
company production in to (possibly) two new productions, `CT` and `FP`, via the rule `CP -> CT FP| FP | CT`.

`CT` is a production that will capture a company type description and has the
following production rule: `CT -> CT '{word}' | '{word}'`, with word being a keyword from the set described above. The subtree
thus created can be used to obtain a list of company keywords to use in the search. This seems overly complicated, but in
fact our index is not expected to change much over time and these rules can be programatically generated in python, so in fact
this is a one-liner. Since a query is expected to only match a small subset of all such keywords, query parsing won't have 
significant overhead either, and the resulting syntax tree can still be manageable.

`FP` is a Field Production, and is responsible for matching free text to relevant company fields (for the time being just
number of employees, revenue, region and country). The rule is as follows: `FP -> FPC FP | FPC`. Basically, we either just
generate one field condition (`FPC` branch), or a sequence of field conditions (`FPC FP` branch).

`FPC` stand for Field Production Condition, and is responsible for identifying actual fields that we want to understand. The
production rule is `FPC -> REV | EMP | LOC`, i.e. we expect to find a condition on revenue, employees or location. 

Each of the `REV`, `EMP` and `LOC` productions follow the same principles, so we only explain `EMP` as an example.
`EMP` has the following production rules: `EMP -> CO 'CARDINAL' 'employees' | 'employees' CO 'CARDINAL'`. Basically,
an employee field is expected to be of the form comparator, cardinal and the 'employees' word (in different orders).
Notice that since we've preprocessed our input query, the potentially very varied cardinality indicator has been replaced by a
simple placeholder called 'CARDINAL'. `CO` is a simple production rule that's a stand-in for a comparator.

#### Putting it all together

Now we went from a free text query to a relatively easy to understand syntax tree. This tree can be processed to obtain values for
respective fields of interest (e.g. an `EMP` node is known to refer to an employee field). We now just need to replace the 
placeholders from the preprocessing steps with their inital values and we are good to go to throw a query to elastic-search and see
what we get.

## TODO

The above is a theoretically elegant solution but has some practical limitations which we should address in the below list

* extend grammar and parsing to handle more than one condition per field where relevant (e.g. revenue between values)
* handle the case where the query fails to parse for whatever reason by falling back to a textual search on all fields.
* investigate allowing some natural flexibility in our grammar (e.g. the query 'telecom companies with more than 5 employees'
would fail because the grammar doesn't know how to handle 'with' yet)
* (STRETCH) logical operators such as and/or/not and properly formulate the query using them
* (STRETCH) perform query expansion in case we have keywords that don't exist in the grammar
