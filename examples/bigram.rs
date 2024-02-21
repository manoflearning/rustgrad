extern crate rustgrad;
use std::fs;
use rustgrad::*;

pub fn stoi(c: char) -> usize {
    if c == '.' { return 0; }
    else { return (c as usize) - ('a' as usize) + 1; }
}
pub fn itos(i: usize) -> char {
    if i == 0 { return '.'; }
    else { return (i as u8 + 'a' as u8 - 1) as char; }
}

pub fn main() {
    let file = fs::read_to_string("data/names.txt").expect("Error opening file");
    let words = file.split("\n").collect::<Vec<&str>>();

    let mut n = torch::zeros(vec![27, 27]);

    for w in words.iter() {
        for i in 0..w.len() - 1 {
            let ix1 = (w.chars().nth(i).unwrap() as usize) - ('a' as usize) + 1;
            let ix2 = (w.chars().nth(i + 1).unwrap() as usize) - ('a' as usize) + 1;
            
            n.data[[ix1, ix2]] += 1.;
            if i == 0 { n.data[[0, ix1]] += 1.; }
            if i + 1 == w.len() - 1 { n.data[[ix2, 0]] += 1.; }
        }
    }

    let mut out: String = String::new();
    let mut ix = 0;
    loop {
        let mut n0 = n.slice(ix..ix+1, 0);
        n0 = n0.clone() / n0.sum();
        ix = torch::multinomial(&n0, 1, true).item() as usize;
        out.push(itos(ix));
        if ix == 0 { break; }
    }
    println!("{}", out);
}