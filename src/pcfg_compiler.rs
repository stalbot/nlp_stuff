use common::*;

// use rustc_serialize::Encodable;
use rustc_serialize::json;


pub fn parse_pcfg(json_pcfg: &str) -> Pcfg {
    json::decode(json_pcfg).expect("Should parse JSON, otherwise boom")
}

pub fn parse_lexicon(lex_pcfg: &str) -> Pcfg {
    json::decode(lex_pcfg).expect("Should parse JSON, otherwise boom")
}

#[cfg(test)]
mod test_pcfg_compiler {
    #[test]
    fn can_parse_pcfg() {
        parse_pcfg(
r#"
{
    "
}
"#
        )
    }

    #[test]
    fn can_parse_pcfg() {
    }
}
