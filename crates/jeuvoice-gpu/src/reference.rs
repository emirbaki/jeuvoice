//! Reads reference tensors that `tools/reference/*.py` writes (`.safetensors`, f32).
//! Tests compare kernel results with them.

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

/// Every tensor in the file as `(shape, f32 values)`, by name.
pub fn load(path: &Path) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("{} is not a safetensors file: {e}", path.display()))?;
    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != Dtype::F32 {
            return Err(format!("{name} is {:?}. Reference files hold f32 only", view.dtype()));
        }
        let values = view.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();
        out.insert(name, (view.shape().to_vec(), values));
    }
    Ok(out)
}
