use std::cmp::{Ord, Ordering};
use std::collections::{BTreeMap, HashMap};

use std::rc::Rc;

pub const START_STATE_LABEL : &'static str = "$S";
pub const MINIMUM_ABSOLUTE_PROB : f32 = 0.00001;
pub const MINIMUM_PROB_RATIO : f32 = 0.01;
pub const PARENT_PROB_PENALTY : f32 = 0.9;

#[derive(Debug)]
pub struct PCFGProduction<'a> {
    pub elements: Vec<&'a PcfgEntry<'a>>,
    pub count: f32,
    pub head: Option<usize>,
    pub full_features: Vec<&'a str>
}

#[derive(Debug, Clone)]
pub struct SemanticEntry {
}

#[derive(Debug)]
pub struct ParseState<'a> {
    pub prob: f32,
    pub features: Features<'a>,
    pub node_stack: Vec<ParseNode<'a>>,
    pub prior_parse: Vec<ParsedToken<'a>>,
    pub semantics: SemanticEntry
}

#[derive(Debug, Clone)]
pub struct ParsedToken<'a> {
    pub word: &'a BareWord,
    pub features: Features<'a>,
    pub pos: PartOfSpeech
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

#[derive(Debug, Clone)]
pub struct ParseNode<'a> {
    pub label: &'a GrammarLabel,
    pub num_children: usize,
    pub production: Option<&'a PCFGProduction<'a>>
}

impl<'a> ParseNode<'a> {
    pub fn is_head(&self) -> bool {
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
pub struct Lemma {
    pub count: f32
}

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy)]
pub enum PartOfSpeech {
    // Token is for filler words that fall directly out of grammar
    Noun, Verb, Adj, Adv, Token
}
impl PartOfSpeech {
    pub fn into_label(&self) -> &'static str {
        match *self {
            PartOfSpeech::Noun => "$N",
            PartOfSpeech::Verb => "$V",
            PartOfSpeech::Adj => "$A",
            PartOfSpeech::Adv => "$R",
            PartOfSpeech::Token => "$TOK"
        }
    }

    pub fn from_label(label: &str) -> Option<PartOfSpeech> {
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
    pub features: Features<'a>,
    pub lemmas: HashMap<&'a LemmaName, Lemma>,
    pub total_count: f32,
    pub pos: PartOfSpeech,
    pub name: &'a SynsetName
}

#[derive(Debug)]
pub struct GlobalParseData<'a> {
    pub pcfg: Pcfg<'a>,
    pub synset_lkup: SynsetLkup<'a>,
    pub lexical_kup: LexicalLkup<'a>
}

#[derive(Debug)]
pub struct PcfgEntry<'a> {
    pub parents_total_count: f32,
    pub parents: HashMap<(&'a GrammarLabel, usize), f32>,
    pub label: &'a GrammarLabel,
    pub productions_count_total: f32,
    pub productions: Vec<PCFGProduction<'a>>,
    pub is_lex_node: bool,
    pub features: Features<'a>
}

pub type LexicalLkup<'a> = HashMap<&'a BareWord, HashMap<&'a SynsetName, f32>>;
pub type SynsetLkup<'a> = HashMap<&'a SynsetName, Synset<'a>>;
pub type Pcfg<'a> = HashMap<&'a GrammarLabel, PcfgEntry<'a>>;

#[derive(Hash, Debug, Eq, PartialEq, Clone)]
pub struct Features<'a> (
    // requires hashability
    BTreeMap<&'a str, &'a str>
);

impl<'a> Features<'a> {
    pub fn features_match(&self, f2: &Features) -> bool {
        self.0.iter().all(|(k1, v1)|
            match f2.0.get(k1) {
                Some(v2) => v2 == v1,
                None => true
            }
        )
    }

    pub fn merge_mut(&mut self, features: &Features<'a>) {
        for (k, v) in &features.0 {
            self.0.insert(k, v);
        }
    }

    pub fn has_feature(&self, feature_name : &'a str) -> bool {
        self.0.contains_key(feature_name)
    }

    pub fn get_feature_value(&self, feature_name : &'a str) -> &'a str {
        self.0[feature_name]
    }

    pub fn add_feature(&mut self, feature_name : &'a str, feature_val : &'a str) {
        self.0.insert(feature_name, feature_val);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}
