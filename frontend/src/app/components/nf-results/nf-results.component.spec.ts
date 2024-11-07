import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NfResultsComponent } from './nf-results.component';

describe('NfResultsComponent', () => {
  let component: NfResultsComponent;
  let fixture: ComponentFixture<NfResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NfResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(NfResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
