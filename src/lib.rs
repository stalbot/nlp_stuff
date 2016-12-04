use std::collections::{BinaryHeap, BTreeMap, HashMap};
use std::iter::FromIterator;
use std::ops::Index;
use std::cmp::{Ord, Ordering};
use std::rc::Rc;

extern crate itertools;
use itertools::Itertools;

extern crate rayon;
use rayon::prelude::*;

const START_STATE_LABEL : &'static str = "$S";
const MINIMUM_ABSOLUTE_PROB : f32 = 0.00001;
const MINIMUM_PROB_RATIO : f32 = 0.01;
const PARENT_PROB_PENALTY : f32 = 0.9;

#[derive(Debug)]
struct PCFGProduction<'a> {
    elements: Vec<&'a PcfgEntry<'a>>,
    count: f32,
    head: Option<usize>,
    full_features: Vec<&'a str>
}

#[derive(Debug, Clone)]
struct SemanticEntry {
}

#[derive(Debug)]
pub struct ParseState<'a> {
    prob: f32,
    features: Features<'a>,
    node_stack: Vec<ParseNode<'a>>,
    prior_parse: Vec<ParsedToken<'a>>,
    semantics: SemanticEntry
}

#[derive(Debug, Clone)]
struct TraversingParseState<'a> {
    prob: f32,
    features: Rc<Features<'a>>,
    node_stack: Rc<Vec<ParseNode<'a>>>,
    semantics: Rc<SemanticEntry>,
    prior_parse: Rc<Vec<ParsedToken<'a>>>
}

impl<'a> ParseState<'a> {
    fn to_traversable(self) -> TraversingParseState<'a> {
        TraversingParseState {
            prob: self.prob,
            features: Rc::new(self.features),
            node_stack: Rc::new(self.node_stack),
            semantics: Rc::new(self.semantics),
            prior_parse: Rc::new(self.prior_parse)
        }
    }

    fn with_reversed_stack(mut self) -> ParseState<'a> {
        self.node_stack.reverse();
        self
    }
}

impl<'a> TraversingParseState<'a> {
    fn to_stable(self) -> ParseState<'a> {
        ParseState {
            prob: self.prob,
            features: match Rc::try_unwrap(self.features) {
                Ok(features) => features,
                Err(rc) => (*rc).clone()
            },
            node_stack: match Rc::try_unwrap(self.node_stack) {
                Ok(node_stack) => node_stack,
                Err(rc) => (*rc).clone()
            },
            semantics: match Rc::try_unwrap(self.semantics) {
                Ok(semantics) => semantics,
                Err(rc) => (*rc).clone()
            },
            prior_parse: match Rc::try_unwrap(self.prior_parse) {
                Ok(prior_parse) => prior_parse,
                Err(rc) => (*rc).clone()
            }
        }
    }

    fn with_next_node(mut self,
                      top_node: &ParseNode<'a>,
                      next_label: &'a GrammarLabel,
                      production: Option<&'a PCFGProduction>)
                      -> TraversingParseState<'a> {
        {
            let stack = Rc::make_mut(&mut self.node_stack);
            let mut new_top = top_node.clone();
            new_top.num_children += 1;

            stack.push(new_top);
            stack.push(ParseNode {
                label: next_label,
                production: production,
                num_children: 0
            });
        }
        self
    }
}

#[derive(Debug, Clone)]
struct ParsedToken<'a> {
    word: &'a BareWord,
    features: Features<'a>,
    pos: PartOfSpeech
}

impl<'a> Ord for ParseState<'a> {
    fn cmp(&self, other: &ParseState) -> Ordering {
        self.prob
            .partial_cmp(&other.prob)
            .unwrap_or(Ordering::Equal)
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
impl<'a> Eq for ParseState<'a> {}

// TODO: clobber this duplication with a macro
impl<'a> Ord for TraversingParseState<'a> {
    fn cmp(&self, other: &TraversingParseState) -> Ordering {
        self.prob
            .partial_cmp(&other.prob)
            .unwrap_or(Ordering::Equal)
    }
}

impl<'a> PartialOrd for TraversingParseState<'a> {
    fn partial_cmp(&self, other: &TraversingParseState) -> Option<Ordering> {
        self.prob.partial_cmp(&other.prob)
    }
}

impl<'a> PartialEq for TraversingParseState<'a> {
    fn eq(&self, other: &TraversingParseState) -> bool {
        self.prob == other.prob
    }
}
impl<'a> Eq for TraversingParseState<'a> {}

#[derive(Debug, Clone)]
struct ParseNode<'a> {
    label: &'a GrammarLabel,
    num_children: usize,
    production: Option<&'a PCFGProduction<'a>>
}

impl<'a> ParseNode<'a> {
    fn is_head(&self) -> bool {
        if let Some(production) = self.production {
            if let Some(head_index) = production.head {
                head_index == self.num_children - 1
            } else {
                self.num_children == production.elements.len()
            }
        }
        else {
            // Not sure if this case would ever make it here
            true
        }
    }
}

pub type LemmaName = str;
pub type SynsetName = str;
pub type BareWord = str;
pub type GrammarLabel = str;

#[derive(Debug)]
struct Lemma {
    count: f32
}

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
enum PartOfSpeech {
    // Token is for filler words that fall directly out of grammar
    Noun, Verb, Adj, Adv, Token
}
impl PartOfSpeech {
    fn into_label(&self) -> &'static str {
        match *self {
            PartOfSpeech::Noun => "$N",
            PartOfSpeech::Verb => "$V",
            PartOfSpeech::Adj => "$A",
            PartOfSpeech::Adv => "$R",
            PartOfSpeech::Token => "$TOK"
        }
    }

    fn from_label(label: &str) -> Option<PartOfSpeech> {
        match label {
            "$N" => Some(PartOfSpeech::Noun),
            "$V" => Some(PartOfSpeech::Verb),
            "$A" => Some(PartOfSpeech::Adj),
            "$R" => Some(PartOfSpeech::Adv),
            _ => None
        }
    }
}

#[derive(Debug)]
pub struct Synset<'a> {
    features: Features<'a>,
    lemmas: HashMap<&'a LemmaName, Lemma>,
    total_count: f32,
    pos: PartOfSpeech,
    name: &'a SynsetName
}

#[derive(Debug)]
pub struct GlobalParseData<'a> {
    pcfg: Pcfg<'a>,
    synset_lkup: SynsetLkup<'a>,
    lexical_kup: LexicalLkup<'a>
}

#[derive(Debug)]
pub struct PcfgEntry<'a> {
    parents_total_count: f32,
    parents: HashMap<(&'a GrammarLabel, usize), f32>,
    label: &'a GrammarLabel,
    productions_count_total: f32,
    productions: Vec<PCFGProduction<'a>>,
    is_lex_node: bool,
    features: Features<'a>
}

pub type LexicalLkup<'a> = HashMap<&'a BareWord, HashMap<&'a SynsetName, f32>>;
pub type SynsetLkup<'a> = HashMap<&'a SynsetName, Synset<'a>>;
pub type Pcfg<'a> = HashMap<&'a GrammarLabel, PcfgEntry<'a>>;

#[derive(Hash, Debug, Eq, PartialEq, Clone)]
struct Features<'a> (
    // requires hashability
    BTreeMap<&'a str, &'a str>
);

impl<'a> Features<'a> {
    fn features_match(&self, f2: &Features) -> bool {
        self.0.iter().all(|(k1, v1)|
            match f2.0.get(k1) {
                Some(v2) => v2 == v1,
                None => true
            }
        )
    }

    fn merge_mut(&mut self, features: &Features<'a>) {
        for (k, v) in &features.0 {
            self.0.insert(k, v);
        }
    }
}



fn synsets_split_by_function<'a, 'b>(
        global_data: &'b GlobalParseData<'a>, word: &'a BareWord)
        -> Vec<(&'b Features<'a>, PartOfSpeech, HashMap<&'a str, f32>, f32)> {
    if let Some(lemma_counts) = global_data.lexical_kup.get(word) {
        let mut grouper = HashMap::new();
        let total_count = 0.0;

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
                        weighted_synsets: &HashMap<&'a SynsetName, f32>,
                        existing_sem: &mut SemanticEntry
                        ) -> f32 {
    // TODO: obviously implement this!
    1.0
}

fn get_start_sem() -> SemanticEntry {
    SemanticEntry {
        // TODO!
    }
}

fn create_first_states<'a, 'b>(global_data: &'b GlobalParseData<'a>,
                               word: &'a BareWord)
                               -> BinaryHeap<TraversingParseState<'b>> {
    let synsets_by_function = synsets_split_by_function(global_data, word);

    synsets_by_function.into_iter()
                       .map(|(features, pos, weighted_synsets, prob)| {
        let mut start_sem = get_start_sem();
        let sem_adj_prob = sem_for_lex_node(global_data,
                                            &weighted_synsets,
                                            &mut start_sem);
        ParseState {
            prob: prob * sem_adj_prob,
            semantics: start_sem,
            features: features.clone(),
            node_stack: vec![
                ParseNode {
                    label: pos.into_label(),
                    num_children: 1,
                    production: None
                }
            ],
            prior_parse: vec![
                ParsedToken {
                    word: word,
                    features: features.clone(),
                    pos: pos
                }
            ]
        }.to_traversable()
    }).collect()
}

fn make_next_initial_state<'a, 'b>(pcfg: &'a Pcfg,
                                   parent_label: &'a GrammarLabel,
                                   mut parse_state: TraversingParseState<'a>,
                                   production: &'a PCFGProduction,
                                   new_prob: f32)
                                   -> TraversingParseState<'a> {
    Rc::make_mut(&mut parse_state.node_stack).push(ParseNode {
        production: Some(production),
        num_children: 1,
        label: parent_label
    });
    TraversingParseState {
        semantics: parse_state.semantics,
        // TODO not fully right, needs :isolate_features junk
        features: parse_state.features,
        prob: new_prob,
        prior_parse: parse_state.prior_parse,
        node_stack: parse_state.node_stack
    }
}

fn parents_with_normed_probs<'a>(pcfg: &'a Pcfg, label: &'a GrammarLabel)
        -> Vec<(&'a GrammarLabel, &'a PCFGProduction<'a>, f32)> {
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
                                         -> Vec<ParseState<'b>> {
    let mut found_states = BinaryHeap::new();
    let mut frontier = create_first_states(global_data, word);
    while found_states.len() < beam_size && !frontier.is_empty() {
        // this value will be not-empty b/c of the while()
        let mut parse_state_opt = frontier.pop();

        let state_parents = {
            let parse_state = parse_state_opt.as_ref().unwrap();
            // guaranteed node_stack not-empty b/c always populated in create_first_states
            let current_label = parse_state.node_stack.last().unwrap().label;
            parents_with_normed_probs(&global_data.pcfg, current_label)
        };

        // Jump through some hoops so that on the last iteration of the
        // loop, there is one owned parse_state, and any Rc in it with just a single
        // reference can be mutated directly, rather than copied
        for (ii, &(parent_label, production, prob))
                in state_parents.iter().enumerate() {
            let parse_state = parse_state_opt.unwrap();
            parse_state_opt = if ii == state_parents.len() - 1 {
                Some(parse_state.clone())
            } else {
                None
            };

            let absolute_prob = prob * parse_state.prob;
            if absolute_prob < MINIMUM_ABSOLUTE_PROB {
                continue;
            }

            let parent_state = make_next_initial_state(&global_data.pcfg,
                                                       parent_label,
                                                       parse_state,
                                                       production,
                                                       absolute_prob);
            if parent_label == START_STATE_LABEL {
                found_states.push(parent_state);
            } else {
                frontier.push(parent_state);
            }
        }
    }
    found_states.into_iter().map(|state| {
        // For perf, we built our stacks in the wrong order
        state.to_stable().with_reversed_stack()
    }).collect()
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
    update_state_probs_for_word(global_data,
                                next_possible_states,
                                word,
                                beam_size)
}

pub fn parse_sentence_fragment<'a>(global_data: &'a GlobalParseData<'a>,
                                   sentence: Vec<&'a BareWord>,
                                   beam_size: usize)
                                   -> Vec<ParseState<'a>> {
    let first_word = sentence.first().expect("fragment not empty");
    let first_states = infer_initial_possible_states(global_data,
                                                     first_word,
                                                     beam_size);

    sentence[1 ..].iter().fold(first_states,
                               |current_states, word|
        parse_word(global_data, current_states, word, beam_size)
    )
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
    let mut new_states = vec![];
    states.into_par_iter().map(|state|
        infer_possible_states(global_data, state, beam_size, word_posses)
    ).collect_into(&mut new_states);

    let mut new_states = new_states.into_iter()
                                   .flat_map(|s| s)
                                   .sorted_by(|s1, s2|
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
    let mut best_prob_opt : Option<f32> = None;
    frontier.push(state.to_traversable());

    while let Some(current_state) = frontier.pop() {
        if found.len() >= beam_size {
            break;
        }

        // We stop collecting if the best intermediate state we've got is less
        // than X times as likely as the best we've found
        // TODO: this could be more efficient if checked b4 put on frontier
        if let Some(best_prob) = best_prob_opt {
            if current_state.prob / best_prob < MINIMUM_PROB_RATIO {
                break;
            }
        } else {
            best_prob_opt = found.first().map(|st : &TraversingParseState|
                                              st.prob);
        }

        get_successor_states(
            global_data,
            current_state,
            &mut found,
            &mut frontier,
            word_posses
        )
    }
    found.into_iter().map(|f| f.to_stable()).collect()
}

fn get_parent_state<'a>(mut state: TraversingParseState<'a>)
                        -> Option<TraversingParseState<'a>> {
    if let Some(_) = Rc::make_mut(&mut state.node_stack).pop() {
        Some(state)
    } else {
        None
    }
}

fn is_term_sym(label: &GrammarLabel) -> bool {
   !label.starts_with("$")
}

fn get_successor_states<'a, 'b>(global_data: &'a GlobalParseData,
                                mut state: TraversingParseState<'a>,
                                found: &mut Vec<TraversingParseState<'a>>,
                                frontier: &mut BinaryHeap<TraversingParseState<'a>>,
                                word_posses: &'b Vec<PartOfSpeech>)
                                -> ()
{
    // Can just return if this is None
    let top_node = match Rc::make_mut(&mut state.node_stack).pop() {
        Some(top_node) => top_node,
        None => return
    };
    let prod_child_count = top_node.production
                                   .map(|prod| prod.elements.len())
                                   .unwrap_or(0);
    if top_node.num_children >= prod_child_count {
        if let Some(parent_state) = get_parent_state(state) {
            frontier.push(parent_state);
        }
    } else {
        get_successor_child_states(global_data,
                                   state,
                                   found,
                                   frontier,
                                   word_posses,
                                   top_node);
    }
}

fn set_next_features<'a>(state: &mut TraversingParseState<'a>,
                         next_entry: &PcfgEntry<'a>,
                         current_node: &ParseNode<'a>) -> () {
    let previous_features = state.prior_parse
                                 .last()
                                 .map(|p| &p.features);
    let features = Rc::make_mut(&mut state.features);
    if !current_node.is_head() {
        features.0.clear();
    }
    if let Some(previous_features) = previous_features {
        if let Some(full_features) = current_node.production
                                                 .map(|p| &p.full_features) {
            for feature_name in full_features {
                if !features.0.contains_key(feature_name)
                        && previous_features.0.contains_key(feature_name) {
                    features.0.insert(feature_name,
                                      previous_features.0[feature_name]);
                }
            }
        }
    }
    features.merge_mut(&next_entry.features);
}

fn set_next_semantics<'a>(global_data: &'a GlobalParseData,
                          state: &mut TraversingParseState) -> () {
    // TODO!
    ()
}

fn get_successor_child_states<'a, 'b>(global_data: &'a GlobalParseData,
                                      mut state: TraversingParseState<'a>,
                                      found: &mut Vec<TraversingParseState<'a>>,
                                      frontier: &mut BinaryHeap<TraversingParseState<'a>>,
                                      word_posses: &'b Vec<PartOfSpeech>,
                                      top_node: ParseNode<'a>)
                                      -> () {
    let num_children = top_node.num_children;
    let next_production = &top_node.production.expect("b/c otherwise 0 above, this isn't amazing style though");
    let next_entry = next_production.elements[num_children];
    let next_label = next_entry.label;
    set_next_features(&mut state, next_entry, &top_node);
    set_next_semantics(global_data, &mut state);

    if next_entry.is_lex_node || is_term_sym(next_label) {
        let pos_for_label = PartOfSpeech::from_label(next_label);
        if let Some(pos) = pos_for_label {
            if word_posses.contains(&pos) || word_posses.is_empty() {
                found.push(state.with_next_node(&top_node, next_label, None));
            }
        } else {
            found.push(state.with_next_node(&top_node, next_label, None));
        }
    } else {
        add_next_intermediate_states(state, frontier, next_entry, top_node);
    }
}

fn add_next_intermediate_states<'a>(state: TraversingParseState<'a>,
                                    frontier: &mut BinaryHeap<TraversingParseState<'a>>,
                                    next_entry: &'a PcfgEntry,
                                    top_node: ParseNode<'a>) -> () {
    let new_productions = &next_entry.productions;
    // Same trick again, provide the mutable state for re-use in the last
    // iteration of the loop
    let mut parse_state_opt = Some(state);
    let last_index = new_productions.len() - 1;
    for (ii, production) in new_productions.iter().enumerate() {
        let mut state = parse_state_opt.unwrap();
        parse_state_opt = if ii == last_index {
            Some(state.clone())
        } else {
            None
        };

        state.prob *= (production.count / next_entry.productions_count_total) *
                       PARENT_PROB_PENALTY;
        if state.prob >= MINIMUM_ABSOLUTE_PROB {
            frontier.push(state.with_next_node(&top_node,
                                               next_entry.label,
                                               Some(production)));
        }
    }
}

fn update_state_probs_with_lex_node<'a, 'b>(
        global_data: &'a GlobalParseData,
        mut state: TraversingParseState<'a>,
        synset_info: &(&'b Features<'a>, PartOfSpeech, HashMap<&'a str, f32>, f32),
        word: &'a BareWord)
        -> Option<TraversingParseState<'a>>  {
    let label = state.node_stack.last().unwrap().label;
    let &(features, pos, ref syns_to_probs, prob_adj) = synset_info;

    if label == pos.into_label() && features.features_match(&state.features) {
        let sem_adj_prob = sem_for_lex_node(global_data,
                                            syns_to_probs,
                                            Rc::make_mut(&mut state.semantics));
        state.prob *= sem_adj_prob * prob_adj;
        Rc::make_mut(&mut state.prior_parse).push(
            ParsedToken {
                word: word,
                features: features.clone(),
                pos: pos
            }
        );
        Some(state)
    } else {
        None
    }
}

fn check_state_against_synsets<'a, 'b>(
        global_data: &'a GlobalParseData,
        state: ParseState<'a>,
        synsets_info: &Vec<(&'b Features<'a>, PartOfSpeech, HashMap<&'a str, f32>, f32)>,
        word: &'a BareWord)
        -> Vec<TraversingParseState<'a>>  {
    // All states at this point have nodes
    let label = state.node_stack.last().unwrap().label;
    let mut state = state.to_traversable();

    if is_term_sym(label) && label == word {
        Rc::make_mut(&mut state.prior_parse).push(
            ParsedToken {
                word: word,
                features: (*state.features).clone(),
                pos: PartOfSpeech::Token
            }
        );
        vec![state]
    } else {
        // And one more time with this nonsense trick
        let mut states_for_ret = vec![];
        let mut parse_state_opt = Some(state);
        for (ii, info) in synsets_info.iter().enumerate() {
            let state = parse_state_opt.unwrap();
            parse_state_opt = if ii == synsets_info.len() - 1 {
                Some(state.clone())
            } else {
                None
            };
            if let Some(state) = update_state_probs_with_lex_node(global_data,
                                                                  state,
                                                                  info,
                                                                  word) {
                states_for_ret.push(state);
            }
        }
        states_for_ret
    }
}

fn update_state_probs_for_word<'a>(global_data: &'a GlobalParseData,
                                   states: Vec<ParseState<'a>>,
                                   word: &'a BareWord,
                                   beam_size: usize)
                                   -> Vec<ParseState<'a>> {
    let synsets_info = synsets_split_by_function(global_data, word);

    let mut updated_states = states.into_iter().flat_map(|state|
        check_state_against_synsets(global_data, state, &synsets_info, word)
    ).sorted_by(|s1, s2|
        s1.prob.partial_cmp(&s2.prob).expect("probs should compare")
    );
    updated_states.truncate(beam_size);

    let prob_total = updated_states.iter().map(|s| s.prob).sum();

    updated_states.into_iter().map(|mut state| {
        state.prob /= prob_total;
        state.to_stable()
    }).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
