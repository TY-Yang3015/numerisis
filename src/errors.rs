#[derive(Debug)]
#[allow(dead_code)]
pub enum ArrayCheckError {
    EmptyArray,
    AllZeros,
    InvalidArray,
}

#[derive(Debug)]
pub enum UnparallelizableError {
    Unparallelizable,
}
