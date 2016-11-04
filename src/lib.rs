use std::collections::BinaryHeap;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::cmp::Ord;
use std::cmp::Ordering;
use std::ops::{Add, Mul};

extern crate itertools;
use itertools::Itertools;

const START_STATE_LABEL : &'static str = "$S";
const MINIMUM_ABSOLUTE_PROB : f32 = 0.00001;

#[derive(Debug)]
struct PCFGProduction {
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
struct ParseNode<'a> {
    label: &'a GrammarLabel,
    production: Option<&'a PCFGProduction>,
    children: Vec<ParseNode<'a>>,
    features: Features<'a>
}

type LemmaName = str;
type SynsetName = str;
type BareWord = str;
type GrammarLabel = str;

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
struct PcfgEntry<'a> {
    parents_total_count: f32,
    parents: HashMap<(&'a GrammarLabel, usize), f32>,
    productions: Vec<PCFGProduction>
}

type LexicalLkup<'a> = HashMap<&'a BareWord, HashMap<&'a SynsetName, f32>>;
type SynsetLkup<'a> = HashMap<&'a SynsetName, Synset<'a>>;
type Pcfg<'a> = HashMap<&'a GrammarLabel, PcfgEntry<'a>>;

#[derive(Hash, Debug, Eq, PartialEq, Clone)]
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
                                  features: features.clone(),
                                  production: None,
                                  children: Vec::new() };

        ParseState {
            prob: prob,
            semantics: start_sem,
            node: ParseNode {
                label: pos.into_label(),
                features: features.clone(),
                production: None,
                children: vec![lex_node]
            }
        }
    }).collect()
}

fn make_next_initial_state<'a, 'b>(pcfg: &'a Pcfg,
                                   parent_label: &'a GrammarLabel,
                                   parse_state: &'b ParseState<'a>,
                                   production: &'a PCFGProduction,
                                   new_prob: f32)
                                   -> ParseState<'a> {
    ParseState{
        semantics: parse_state.semantics.clone(),
        prob: new_prob,
        node: ParseNode {
            label: parent_label,
            production: Some(production),
            children: vec![parse_state.node.clone()],
            features: parse_state.node.features.clone() // TODO not fully right
        }
    }
}

fn parents_with_normed_probs<'a>(pcfg: &'a Pcfg, label: &'a GrammarLabel)
        -> Vec<(&'a GrammarLabel, &'a PCFGProduction, f32)> {
    // it is a programmer error to have an invalid label leak in here
    let pcfg_entry = pcfg.get(label).unwrap();
    let mut parents = pcfg_entry.parents
                                .iter()
                                .map(|(&(parent_label, prod_index), &count)| {
        let normalized_count = count / pcfg_entry.parents_total_count;
        let production = pcfg.get(parent_label)
                             .unwrap()
                             .productions
                             .get(prod_index)
                             .unwrap();
        (parent_label, production, normalized_count)
    }).collect::<Vec<_>>();
    parents.sort_by(|&(_, _, c1), &(_, _, c2)|
        c1.partial_cmp(&c2).expect("probs should compare")
    );
    parents
}

fn infer_initial_possible_states<'a, 'b>(global_data: &'b GlobalParseData<'a>,
                                         word: &'a BareWord,
                                         beam_size: usize)
                                         -> BinaryHeap<ParseState<'b>> {
    let mut found_states = BinaryHeap::new();
    let mut frontier = create_first_states(global_data, word);
    while found_states.len() < beam_size && !frontier.is_empty() {
        let parse_state = frontier.pop().unwrap(); // not-empty b/c while()
        let current_label = parse_state.node.label;
        let state_parents = parents_with_normed_probs(&global_data.pcfg,
                                                      current_label);

        for (parent_label, production, prob) in state_parents {
            let absolute_prob = prob * parse_state.prob;
            if absolute_prob < MINIMUM_ABSOLUTE_PROB {
                continue;
            }

            let parent_state = make_next_initial_state(&global_data.pcfg,
                                                       parent_label,
                                                       &parse_state,
                                                       production,
                                                       absolute_prob);
            if parent_label == START_STATE_LABEL {
                found_states.push(parent_state);
            } else {
                frontier.push(parent_state);
            }
        }
    }
    found_states
}

fn parse_word<'a>(global_data: &'a GlobalParseData<'a>,
                  states: Vec<ParseState<'a>>,
                  word: &'a BareWord,
                  beam_size: usize)
                  -> Vec<ParseState<'a>> {
    let word_posses = possible_parts_of_speech_for_word(global_data, word);
    let next_possible_states = infer_possible_states_mult(
                                    global_data,
                                    states,
                                    beam_size,
                                    &word_posses);
    update_state_probs_for_word(global_data, next_possible_states, word)
}

fn possible_parts_of_speech_for_word(global_data: &GlobalParseData,
                                     word: &BareWord)
                                     -> Vec<PartOfSpeech> {
    // TODO!
    vec![PartOfSpeech::Noun, PartOfSpeech::Verb, PartOfSpeech::Adj, PartOfSpeech::Adv]
}

fn infer_possible_states_mult<'a, 'b>(global_data: &'a GlobalParseData,
                                      states: Vec<ParseState<'a>>,
                                      beam_size: usize,
                                      word_posses: &'b Vec<PartOfSpeech>)
                                      -> Vec<ParseState<'a>> {
    let new_states = states.into_iter().flat_map(|state|
        infer_possible_states(global_data, state, beam_size, word_posses)
    );
    let mut new_states = new_states.sorted_by(|s1, s2|
        s1.prob.partial_cmp(&s2.prob).expect("probs should compare")
    );
    new_states.truncate(beam_size);
    new_states
}

fn infer_possible_states<'a, 'b>(global_data: &'a GlobalParseData,
                                 state: ParseState<'a>,
                                 beam_size: usize,
                                 word_posses: &'b Vec<PartOfSpeech>)
                                 -> Vec<ParseState<'a>> {
    let mut found = vec![];
    let mut frontier = BinaryHeap::new();
    frontier.push(state);

    while let Some(current_state) = frontier.pop() {
        if found.len() >= beam_size {
            break;
        }

        get_successor_states(
            global_data,
            current_state,
            &mut found,
            &mut frontier,
            word_posses
        )
    }
    vec![]
}

fn get_successor_states<'a, 'b>(global_data: &'a GlobalParseData,
                                state: ParseState<'a>,
                                found: &mut Vec<ParseState<'a>>,
                                frontier: &mut BinaryHeap<ParseState<'a>>,
                                word_posses: &'b Vec<PartOfSpeech>)
                                -> ()
{
    ()
}

fn update_state_probs_for_word<'a>(global_data: &'a GlobalParseData,
                                   states: Vec<ParseState<'a>>,
                                   word: &'a BareWord)
                                   -> Vec<ParseState<'a>> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
