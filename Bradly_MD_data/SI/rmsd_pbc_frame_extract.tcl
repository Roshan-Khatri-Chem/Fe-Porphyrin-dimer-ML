# 1. Load pbctools package
package require pbctools

# 2. Apply PBC wrapping to keep molecules intact
puts "Applying PBC wrapping..."
pbc wrap -compound fragment -center origin -all

# 3. Select Monomer 1 as the reference group (Adjust selection to match your system)
# Example: "resid 1", "fragment 0", or atom indices "index 0 to 45"
set ref_sel [atomselect top "resid 1" frame 0]
set align_sel [atomselect top "resid 1"]
set all_sel [atomselect top "all"]

# 4. Align every frame so Monomer 1 remains fixed in space
set num_frames [molinfo top get numframes]
puts "Aligning Monomer 1 across $num_frames frames..."

for {set i 0} {$i < $num_frames} {incr i} {
    # Move selection to current frame
    $align_sel frame $i
    $all_sel frame $i
    
    # Calculate best-fit transformation matrix to match Monomer 1 in Frame 0
    set trans_mat [measure fit $align_sel $ref_sel]
    
    # Apply transformation to the entire system (moving Monomer 2 around fixed Monomer 1)
    $all_sel move $trans_mat
}

# 5. Save the aligned trajectory geometries
file mkdir pbc_aligned_frames
cd pbc_aligned_frames

puts "Extracting aligned frames..."
for {set i 0} {$i < $num_frames} {incr i} {
    set sel [atomselect top "all" frame $i]
    set filename [format "geom_%d.xyz" $i]

    animate write xyz $filename beg $i end $i sel $sel
    $sel delete

    if {$i % 2000 == 0} {
        puts "Saved $filename ($i / [expr {$num_frames - 1}])"
    }
}

puts "PBC wrapping, Monomer-1 alignment, and extraction complete!"
exit
