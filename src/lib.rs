use std::collections::HashMap;
use std::collections::BTreeMap;
use std::iter::FromIterator;
use std::collections::hash_map::OccupiedEntry;

#[derive(Debug)]
struct PCFGProduction {
}

#[derive(Debug)]
struct SemanticEntry {
}

#[derive(Debug)]
struct ParseNode<'a> {
    label: &'a str,
    production: PCFGProduction,
    children: Vec<ParseNode<'a>>,
    features: HashMap<&'a str, &'a str>,
    semantics: SemanticEntry
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
            let synset : &'b Synset = &global_data.synset_lkup.get(synset_name).unwrap();
            grouper.entry((&synset.features, &synset.pos))
                   .or_insert_with(|| Vec::new())
                   .push((synset, lemma_count))
        }

        grouper.into_iter().map(|((features, pos), synsets_w_counts)| {
            let total_group_count : f32 = synsets_w_counts.iter().map(|&(_, c)| c).sum();
            let weighted_synsets = HashMap::from_iter(
                synsets_w_counts.iter().map(|&(synset, count)|
                    (synset.name, count / total_group_count)
                )
            );

            (features, pos.clone(), weighted_synsets, (total_group_count / total_count))
        }).collect()
    } else {
        Vec::new()
    }
}

fn construct_initial_states() -> i32 {
    12
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
