use std::collections::BinaryHeap;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::collections::hash_map::OccupiedEntry;
use std::cmp::Ord;
use std::cmp::Ordering;

#[derive(Debug)]
struct PCFGProduction {
}

#[derive(Debug)]
struct SemanticEntry {
}

#[derive(Debug)]
struct ParseState<'a> {
    prob: f32,
    node: ParseNode<'a>,
    semantics: SemanticEntry
}

impl<'a> Ord for ParseState<'a> {
    fn cmp(&self, other: &ParseState) -> Ordering {
        self.prob.partial_cmp(&other.prob).unwrap_or(Ordering::Equal)
    }
}

impl<'a> PartialOrd for ParseState<'a> {
    fn partial_cmp(&self, other: &ParseState) -> Option<Ordering> {
        self.prob.partial_cmp(&other.prob)
    }
}

impl<'a> PartialEq for ParseState<'a> {
    fn eq(&self, other: &ParseState) -> bool {
        self.prob == other.prob
    }
}
impl<'a> Eq for ParseState<'a>{}

#[derive(Debug)]
struct ParseNode<'a> {
    label: &'a str,
    production: Option<PCFGProduction>,
    children: Vec<ParseNode<'a>>,
    features: &'a Features<'a>
}

type LemmaName = str;
type SynsetName = str;
type BareWord = str;
type GrammarKey = str;

#[derive(Debug)]
struct Lemma {
    count: f32
}

#[derive(Eq, PartialEq, Hash, Debug, Clone)]
enum PartOfSpeech {
    Noun, Verb, Adj, Adv
}
impl PartOfSpeech {
    fn into_label(&self) -> &'static str {
        match *self {
            PartOfSpeech::Noun => "$N",
            PartOfSpeech::Verb => "$V",
            PartOfSpeech::Adj => "$A",
            PartOfSpeech::Adv => "$R"
        }
    }
}

#[derive(Debug)]
struct Synset<'a> {
    features: Features<'a>,
    lemmas: HashMap<&'a LemmaName, Lemma>,
    total_count: f32,
    pos: PartOfSpeech,
    name: &'a SynsetName
}

#[derive(Debug)]
struct GlobalParseData<'a> {
    pcfg: Pcfg<'a>,
    synset_lkup: SynsetLkup<'a>,
    lexical_kup: LexicalLkup<'a>
}

#[derive(Debug)]
struct PcfgEntry {
}

type LexicalLkup<'a> = HashMap<&'a BareWord, HashMap<&'a SynsetName, f32>>;
type SynsetLkup<'a> = HashMap<&'a SynsetName, Synset<'a>>;
type Pcfg<'a> = HashMap<&'a GrammarKey, PcfgEntry>;

#[derive(Hash, Debug, Eq, PartialEq)]
struct Features<'a> (
    // requires hashability
    BTreeMap<&'a str, &'a str>
);

fn synsets_split_by_function<'a, 'b>(
        global_data: &'b GlobalParseData<'a>, word: &'a BareWord)
        -> Vec<(&'b Features<'a>, PartOfSpeech, HashMap<&'a str, f32>, f32)> {
    if let Some(lemma_counts) = global_data.lexical_kup.get(word) {
        let mut grouper = HashMap::new();
        let mut total_count = 0.0;

        for (synset_name, lemma_count) in lemma_counts {
            let synset : &'b Synset = &global_data.synset_lkup
                                                  .get(synset_name)
                                                  .unwrap();
            // TODO: parity difference, features not merged with lemma features
            grouper.entry((&synset.features, &synset.pos))
                   .or_insert_with(|| Vec::new())
                   .push((synset, lemma_count))
        }

        grouper.into_iter().map(|((features, pos), synsets_w_counts)| {
            let total_group_count : f32 = synsets_w_counts.iter()
                                                          .map(|&(_, c)| c)
                                                          .sum();
            let weighted_synsets = HashMap::from_iter(
                synsets_w_counts.iter().map(|&(synset, count)|
                    (synset.name, count / total_group_count)
                )
            );

            (features,
             pos.clone(),
             weighted_synsets,
             (total_group_count / total_count))
        }).collect()
    } else {
        Vec::new()
    }
}

fn sem_for_lex_node<'a>(global_data: &'a GlobalParseData,
                        weighted_synsets: HashMap<&'a SynsetName, f32>,
                        existing_sem: Option<SemanticEntry>
                        ) -> SemanticEntry {
    // TODO: obviously implement this!
    SemanticEntry{}
}

fn create_first_states<'a, 'b>(global_data: &'b GlobalParseData<'a>,
                               word: &'a BareWord)
                               -> BinaryHeap<ParseState<'b>> {
    let synsets_by_function = synsets_split_by_function(global_data, word);

    synsets_by_function.into_iter()
                       .map(|(features, pos, weighted_synsets, prob)| {
        let start_sem = sem_for_lex_node(global_data, weighted_synsets, None);
        let lex_node = ParseNode{ label: word,
                                  features: features,
                                  production: None,
                                  children: Vec::new() };

        ParseState {
            prob: prob,
            semantics: start_sem,
            node: ParseNode {
                label: pos.into_label(),
                features: features,
                production: None,
                children: vec![lex_node]
            }
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
