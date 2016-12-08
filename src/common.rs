use std::cmp::{Ord, Ordering};
use std::collections::{BTreeMap, HashMap};

pub const START_STATE_LABEL : &'static str = "$S";
pub const MINIMUM_ABSOLUTE_PROB : f32 = 0.00001;
pub const MINIMUM_PROB_RATIO : f32 = 0.01;
pub const PARENT_PROB_PENALTY : f32 = 0.9;

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct PCFGProduction {
    pub count: f32,
    pub head: Option<usize>,
    pub full_features: Vec<String>,
    pub elements: Vec<ProductionElement>
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct ProductionElement {
    pub label: String,
    pub features: Features
}

#[derive(Debug, RustcEncodable, RustcDecodable, Clone)]
pub struct SemanticEntry {
}

#[derive(Debug, RustcEncodable)]
pub struct ParseState<'a> {
    pub prob: f32,
    pub features: Features,
    pub node_stack: Vec<ParseNode<'a>>,
    pub prior_parse: Vec<ParsedToken<'a>>,
    pub semantics: SemanticEntry
}

#[derive(Debug, RustcEncodable, Clone)]
pub struct ParsedToken<'a> {
    pub word: &'a str,
    pub features: Features,
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

#[derive(Debug, RustcEncodable, Clone)]
pub struct ParseNode<'a> {
    pub label: &'a str,
    pub num_children: usize,
    pub production: Option<&'a PCFGProduction>
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

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct Lemma {
    pub count: f32
}

#[derive(Eq, PartialEq, Hash, Debug, RustcEncodable, RustcDecodable, Clone, Copy)]
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

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct Synset {
    pub features: Features,
    pub lemmas: HashMap<String, Lemma>,
    pub total_count: f32,
    pub pos: PartOfSpeech,
    pub name: String
}

#[derive(Debug, RustcEncodable)]
pub struct GlobalParseData {
    pub pcfg: Pcfg,
    pub synset_lkup: SynsetLkup,
    pub lexical_kup: LexicalLkup
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct PcfgEntry {
    pub parents_total_count: f32,
    pub parents: HashMap<(String, usize), f32>,
    pub label: String,
    pub productions_count_total: f32,
    pub productions: Vec<PCFGProduction>,
    pub is_lex_node: bool,
    pub features: Features
}

pub type LexicalLkup = HashMap<String, HashMap<String, f32>>;
pub type SynsetLkup = HashMap<String, Synset>;
pub type Pcfg = HashMap<String, PcfgEntry>;

#[derive(Hash, Debug, RustcEncodable, RustcDecodable, Eq, PartialEq, Clone)]
pub struct Features (
    // requires hashability
    BTreeMap<String, String>
);

impl<'a> Features {
    pub fn features_match(&self, f2: &Features) -> bool {
        self.0.iter().all(|(k1, v1)|
            match f2.0.get(k1) {
                Some(v2) => v2 == v1,
                None => true
            }
        )
    }

    pub fn merge_mut(&mut self, features: &Features) {
        for (k, v) in &features.0 {
            self.0.insert(k.clone(), v.clone());
        }
    }

    pub fn has_feature(&self, feature_name : &'a str) -> bool {
        self.0.contains_key(feature_name)
    }

    pub fn get_feature_value(&'a self, feature_name: &'a str) -> &'a str {
        &self.0[feature_name]
    }

    pub fn add_feature(&mut self, feature_name : &'a str, feature_val : &'a str) {
        self.0.insert(String::from(feature_name), String::from(feature_val));
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}
